from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import torch
import gc
import json


from DIFFUSION.generation_functions import load_checkpoint, generate_sample, generate_sample_ddim, tensor_to_base64
from DIFFUSION.model import CFGDenoiser, TimeEncoding

app = FastAPI()

AVAILABLE_MODELS = ["Diffusion", "Gan", "Vae"]



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
        <h2>Generation Settings</h2>

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
                <input type="number" name="faces_per_row" value="5" min="1" max="10">
                <div class="hint">Min 1, Max 10</div>
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
        b64_img = tensor_to_base64(step_tensor)
        # Yield a JSON line
        data = json.dumps({"status": "generating", "image": b64_img})
        yield data + "\n"

    # Final "Done" message (send the last frame again to be sure)
    yield json.dumps({"status": "complete", "image": b64_img}) + "\n"


@app.post("/stream_feed_json")
async def stream_feed_json(request: Request):
    form_data = await request.form()

    # Extract params (same as before)
    approach = form_data.get('approach')
    cfg_lambda = int(form_data.get('cfg_lambda'))
    faces_per_row = int(form_data.get('faces_per_row'))
    number_of_row = int(form_data.get('number_of_row'))

    classes = {}
    for i in range(number_of_row):
        key = f"row_{i}_class"
        if key in form_data:
            classes[i] = [int(digit) for digit in form_data[key]]

    return StreamingResponse(
        stream_generator_json(classes, approach, cfg_lambda, faces_per_row, number_of_row),
        media_type="application/x-ndjson"
    )
