from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from torchvision import utils
import gc
import json
import torch
import redis_db as db
import os
import uuid
from typing import Optional
from DIFFUSION.generation_functions import load_checkpoint, generate_sample, generate_sample_ddim, tensor_to_base64
from DIFFUSION.model import CFGDenoiser, TimeEncoding

app = FastAPI()
db.init_db()
SAVE_DIR = 'FACES'
os.makedirs(name=SAVE_DIR,exist_ok=True)
AVAILABLE_MODELS = ["Diffusion"]

app.mount(f"/{SAVE_DIR}", StaticFiles(directory=SAVE_DIR), name="images")


def save_images(batch_tensor, labels, approach):
    """Saves the individual images contained in the batch tensors on filesystem,
    and saves the path and meta-data on redis"""
    batch_size = batch_tensor.shape[0]

    for i in range(batch_size):
        img_tensor = batch_tensor[i]

        l_vec = labels[i].cpu().int().numpy()
        gender = int(l_vec[0])
        glasses = int(l_vec[1])
        beard = int(l_vec[2])
        class_type = str(gender) + str(glasses) + str(beard)
        filename = f"{uuid.uuid4().hex}_{class_type}_{approach}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)

        utils.save_image(img_tensor, filepath, normalize=True, value_range=(-1, 1))


        db_path = f"/{SAVE_DIR}/{filename}"
        db.insert_face(db_path, gender, beard, glasses, approach)






# Other generation methodogies (trough VAE and GAN) are not supported
class ModelManager:
    def __init__(self):
        self.diffusion_model = None
        self.time_encoder = None
        self.vae_model = None
        self.gan_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _unload_others(self, keep: str):
        """Helper to clear VRAM of unused models."""
        cleared = False
        if keep != 'diffusion' and self.diffusion_model is not None:
            del self.diffusion_model
            del self.time_encoder
            self.diffusion_model = None
            self.time_encoder = None
            cleared = True
        if keep != 'vae' and self.vae_model is not None:
            del self.vae_model
            self.vae_model = None
            cleared = True
        if keep != 'gan' and self.gan_model is not None:
            del self.gan_model
            self.gan_model = None
            cleared = True
        if cleared:
            gc.collect()
            torch.cuda.empty_cache()

    def get_diffusion(self) -> tuple[CFGDenoiser, TimeEncoding]:
        if self.diffusion_model is None or self.time_encoder is None:
            print('Initializing Diffusion...')
            self._unload_others(keep='diffusion')
            self.diffusion_model = CFGDenoiser().to(self.device)
            self.time_encoder = TimeEncoding(L=1000, dim=256, device=self.device)
            load_checkpoint(denoiser=self.diffusion_model, path="./DIFFUSION/diffusion.pt")

        return self.diffusion_model, self.time_encoder

MODEL_MANAGER = ModelManager()


@app.get("/", response_class=HTMLResponse)
def root():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Face Generation Control Panel</title>
        <style>
            :root { --primary: #2563eb; --bg: #f8fafc; --card: #ffffff; --text: #1e293b; }
            body { font-family: system-ui, -apple-system, sans-serif; background-color: var(--bg); color: var(--text); display: flex; justify-content: center; padding: 2rem; }
            .container { background-color: var(--card); padding: 2rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); width: 100%; max-width: 600px; }
            h2 { margin-top: 0; color: var(--primary); border-bottom: 2px solid #e2e8f0; padding-bottom: 1rem; }
            .form-group { margin-bottom: 1.5rem; }
            label { display: block; font-weight: 600; margin-bottom: 0.5rem; font-size: 0.9rem; }
            input[type="number"], input[type="text"], select { width: 100%; padding: 0.75rem; border: 1px solid #cbd5e1; border-radius: 6px; box-sizing: border-box; font-size: 1rem; }
            .row-config { background-color: #f1f5f9; padding: 1rem; border-radius: 8px; margin-top: 0.5rem; }
            .row-item { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem; }
            .row-item label { margin-bottom: 0; width: 80px; }
            button { background-color: var(--primary); color: white; border: none; padding: 1rem; width: 100%; border-radius: 6px; font-size: 1rem; font-weight: bold; cursor: pointer; transition: background-color 0.2s; }
            button:hover { background-color: #1d4ed8; }
            .hint { font-size: 0.8rem; color: #64748b; margin-top: 0.25rem; }

            /* Spinner */
            .loader {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #2563eb;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 20px auto;
                display: none;
            }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

            /* Stream Display */
            #stream-container { text-align: center; margin-top: 2rem; display: none; }
            #stream-display { max-width: 100%; border-radius: 8px; border: 2px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); display: none; }
            .status { margin-bottom: 0.5rem; color: var(--primary); font-weight: 600; }

            /* Download Button */
            #download-btn {
                display: none;
                margin-top: 1rem;
                background-color: #10b981;
                color: white;
                padding: 10px 20px;
                text-decoration: none;
                border-radius: 6px;
                font-weight: bold;
            }
        </style>
    </head>
    <body>


    <div class="container">
        <h2>Generation Settings <a href="/gallery" class="home-link"> Go to gallery &rarr;</a> </h2>

        <form id="gen-form">
            <div class="form-group">
                <label>Method</label>
                <select name="approach">
                    <option value="ddpm">DDPM</option>
                    <option value="ddim">DDIM</option>
                </select>
            </div>

            <div class="form-group">
                <label>Guidance Lambda</label>
                <input type="number" name="cfg_lambda" value="3" min="-20" max="20">
                <div class="hint">Recommended 3</div>
            </div>

            <div class="form-group">
                <label>Samples per Row</label>
                <input type="number" name="faces_per_row" value="5" min="1" max="5">
                <div class="hint">Min 1, Max 5</div>
            </div>

            <div class="form-group">
                <label>Rows</label>
                <input type="number" name="number_of_row" id="number_of_row" value="1" min="1" max="8" onchange="updateRowInputs()">
                <div class="hint">Min 1, Max 8. (Selecting 8 auto-fills classes)</div>
            </div>

            <div id="row-container" class="row-config"></div>

            <br>
            <button id="generate-btn" type="button" onclick="startStream()">Generate</button>
        </form>

        <div id="stream-container">
            <h3 id="status-text" style="color: #2563eb">Initializing...</h3>

            <div id="loading-spinner" class="loader"></div>
            <img id="stream-display" src="" alt="Generation Stream" />

            <br>
            <a id="download-btn" href="#" download="generated_faces.jpg">Download Image</a>
        </div>
    </div>

    <script>
        const possibleClasses = ['000', '001', '010', '011', '100', '101', '110', '111'];
        const container = document.getElementById('row-container');
        const rowInput = document.getElementById('number_of_row');

        function updateRowInputs() {
            container.innerHTML = '';
            let count = parseInt(rowInput.value);
            if (count < 1) count = 1;
            if (count > 8) count = 8;
            rowInput.value = count;

            const isAutoFill = (count === 8);

            for (let i = 0; i < count; i++) {
                const div = document.createElement('div');
                div.className = 'row-item';
                const label = document.createElement('label');
                label.innerText = `Row ${i + 1}:`;

                const select = document.createElement('select');
                select.name = `row_${i}_class`;

                possibleClasses.forEach((cls, index) => {
                    const option = document.createElement('option');
                    option.value = cls;
                    option.text = cls;
                    if (isAutoFill && index === i) option.selected = true;
                    select.appendChild(option);
                });

                if (isAutoFill) {
                    const hidden = document.createElement('input');
                    hidden.type = 'hidden';
                    hidden.name = `row_${i}_class`;
                    hidden.value = possibleClasses[i];
                    div.appendChild(hidden);
                    select.disabled = true;
                }
                div.appendChild(label);
                div.appendChild(select);
                container.appendChild(div);
            }
        }

        // Global variable to track the active stream
        let currentController = null;

        async function startStream() {
            const form = document.getElementById('gen-form');
            const img = document.getElementById('stream-display');
            const status = document.getElementById('status-text');
            const downloadBtn = document.getElementById('download-btn');
            const streamContainer = document.getElementById('stream-container');
            const genBtn = document.getElementById('generate-btn');
            const spinner = document.getElementById('loading-spinner');

            // --- STOP LOGIC ---
            if (currentController) {
                currentController.abort();
                currentController = null;
                status.innerText = "Stopped by user";
                status.style.color = "#f59e0b";
                spinner.style.display = 'none';
                return;
            }

            // --- START LOGIC ---
            currentController = new AbortController();
            const signal = currentController.signal;

            // UI Updates
            genBtn.innerText = "Stop Generation";
            genBtn.style.backgroundColor = "#ef4444";

            streamContainer.style.display = 'block';
            downloadBtn.style.display = 'none';
            spinner.style.display = 'block';
            img.style.display = 'none';

            status.innerText = "Loading model...";
            status.style.color = "#2563eb";

            const formData = new FormData(form);
            let step = ('ddpm' == formData.get('approach')) ? 1000 : 100;
            let counter = 1;
            let firstFrameReceived = false;

            try {
                const response = await fetch('/stream_feed_json', {
                    method: 'POST',
                    body: formData,
                    signal: signal
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    let lines = buffer.split('\\n');
                    buffer = lines.pop();

                    for (let line of lines) {
                        if (!line.trim()) continue;
                        try {
                            const data = JSON.parse(line);

                            if (!firstFrameReceived) {
                                spinner.style.display = 'none';
                                img.style.display = 'inline-block';
                                firstFrameReceived = true;
                            }

                            img.src = "data:image/jpeg;base64," + data.image;


                            let percent = Math.min(100, Math.floor((counter / step) * 100));
                            status.innerText = `Step ${counter}/${step} (${percent}%)`;

                            counter++;

                            if (data.status === 'complete') {
                                status.innerText = "Generation Complete!";
                                status.style.color = "#10b981";
                                downloadBtn.href = img.src;
                                downloadBtn.style.display = 'inline-block';
                            }
                        } catch (e) {
                            console.error("JSON Parse error", e);
                        }
                    }
                }
            } catch (err) {
                if (err.name === 'AbortError') {
                    console.log('Generation aborted.');
                } else {
                    console.error("Stream error:", err);
                    status.innerText = "Error during generation";
                    status.style.color = "red";
                }
            } finally {
                currentController = null;
                genBtn.innerText = "Generate";
                genBtn.style.backgroundColor = "";
                spinner.style.display = 'none';
            }
        }

        // Initialize rows on load
        updateRowInputs();
    </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


def stream_generator_json(classes, approach, cfg_lambda, faces_per_row, number_of_row):
    noise_size = (faces_per_row * number_of_row, 3, 64, 64)
    labels = torch.zeros((noise_size[0], 3), device=MODEL_MANAGER.device)

    den_net, time_encoder = MODEL_MANAGER.get_diffusion()

    for i in range(len(classes)):
        labels[(i * faces_per_row) : (i + 1) * faces_per_row] = torch.tensor(
            classes[i], dtype=torch.float32, device=MODEL_MANAGER.device
        )

    if approach == 'ddim':
        generator = generate_sample_ddim(den_net, time_encoder, noise_size, labels, cfg_lambda)
    else:
        generator = generate_sample(den_net, time_encoder, noise_size, labels, cfg_lambda) #ddpm

    for step_tensor in generator:
        last_tensor = step_tensor
        b64_img = tensor_to_base64(last_tensor)
        # Yield a JSON line
        data = json.dumps({"status": "generating", "image": b64_img})
        yield data + "\n"

    save_images(last_tensor, labels, approach)
    yield json.dumps({"status": "complete", "image": b64_img}) + "\n"


@app.post("/stream_feed_json")
async def stream_feed_json(request: Request):
    form_data = await request.form()

    # Extract params
    approach = form_data.get('approach')
    cfg_lambda = int(form_data.get('cfg_lambda'))
    faces_per_row = int(form_data.get('faces_per_row'))
    number_of_row = int(form_data.get('number_of_row'))

    faces_per_row = max(1, min(faces_per_row, 5))


    classes = {}
    for i in range(number_of_row):
        key = f"row_{i}_class"
        if key in form_data:
            classes[i] = [int(digit) for digit in form_data[key]]

    return StreamingResponse(
        stream_generator_json(classes, approach, cfg_lambda, faces_per_row, number_of_row),
        media_type="application/x-ndjson"
    )






@app.get("/gallery", response_class=HTMLResponse)
def gallery(
    gender: Optional[str] = Query(None, description="0: Female, 1: Male"),
    beard: Optional[str] = Query(None, description="0: No Beard, 1: Beard"),
    glasses: Optional[str] = Query(None, description="0: No Glasses, 1: Glasses"),
    approach: Optional[str] = Query(None, description="ddpm or ddim")
):
    """
    Displays a gallery of generated images with filtering options.
    Accepts strings to handle empty form values (e.g., ?gender=) gracefully.
    """

    gender_int = int(gender) if gender in ['0', '1'] else None
    beard_int = int(beard) if beard in ['0', '1'] else None
    glasses_int = int(glasses) if glasses in ['0', '1'] else None

    if approach is not None and not approach.strip():
        approach = None

    faces = db.get_filtered_faces(
        gender=gender_int,
        beard=beard_int,
        glasses=glasses_int,
        approach=approach,
        n_to_return=50
    )

    # Helper to create option tags
    def mk_opt(val, curr, label):
        sel = "selected" if str(val) == str(curr) else ""
        return f'<option value="{val}" {sel}>{label}</option>'

    # Build HTML for gallery
    images_html = ""
    for face in faces:
        # face['path'] comes from DB as "/FACES/filename.jpg", which matches our mount
        # Metadata
        meta = []
        if face['gender'] == 1: meta.append("Male")
        else: meta.append("Female")
        if face['glasses'] == 1: meta.append("Glasses")
        if face['beard'] == 1: meta.append("Beard")

        meta_str = ", ".join(meta)

        images_html += f"""
        <div class="gallery-item">
            <img src="{face['path']}" loading="lazy" alt="{meta_str}">
            <div class="meta">
                <strong>{face['approach'].upper()}</strong><br>
                <small>{meta_str}</small>
            </div>
        </div>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Generated Gallery</title>
        <style>
            :root {{ --primary: #2563eb; --bg: #f8fafc; --card: #ffffff; }}
            body {{ font-family: system-ui, sans-serif; background: var(--bg); padding: 2rem; }}
            .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; }}
            h1 {{ margin: 0; color: var(--primary); }}
            .controls {{ background: var(--card); padding: 1rem; border-radius: 8px; display: flex; gap: 1rem; flex-wrap: wrap; box-shadow: 0 2px 4px rgb(0 0 0 / 0.1); margin-bottom: 2rem; }}
            select, button {{ padding: 0.5rem; border-radius: 4px; border: 1px solid #ccc; }}
            button {{ background: var(--primary); color: white; border: none; cursor: pointer; font-weight: bold; }}

            .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 1rem; }}
            .gallery-item {{ background: var(--card); border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgb(0 0 0 / 0.1); transition: transform 0.2s; }}
            .gallery-item:hover {{ transform: translateY(-4px); }}
            .gallery-item img {{ width: 100%; height: auto; display: block; }}
            .meta {{ padding: 0.5rem; font-size: 0.8rem; text-align: center; color: #475569; }}
            .home-link {{ text-decoration: none; color: var(--primary); font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Gallery</h1>
            <a href="/" class="home-link">&larr; Back to Generator</a>
        </div>

        <form method="get" class="controls">
            <select name="gender">
                <option value="">Gender (Any)</option>
                {mk_opt(1, gender, 'Male')}
                {mk_opt(0, gender, 'Female')}
            </select>
            <select name="beard">
                <option value="">Beard (Any)</option>
                {mk_opt(1, beard, 'Yes')}
                {mk_opt(0, beard, 'No')}
            </select>
            <select name="glasses">
                <option value="">Glasses (Any)</option>
                {mk_opt(1, glasses, 'Yes')}
                {mk_opt(0, glasses, 'No')}
            </select>
            <select name="approach">
                <option value="">Approach (Any)</option>
                {mk_opt('ddpm', approach, 'DDPM')}
                {mk_opt('ddim', approach, 'DDIM')}
            </select>
            <button type="submit">Filter</button>
            <a href="/gallery" style="align-self:center; margin-left: auto; font-size: 0.9rem;">Reset</a>
        </form>

        <div class="grid">
            {images_html}
        </div>

        { "<p style='text-align:center; color: #666;'>No images found matching these criteria.</p>" if not faces else "" }
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
