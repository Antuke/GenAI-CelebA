import redis
import os
import time

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
SAVE_DIR = "FACES"

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def init_db():
    """Checks connection."""
    try:
        r.ping()
        print(f"Connected to Redis at {REDIS_HOST}")
    except redis.ConnectionError:
        print(f"Error: Could not connect to Redis at {REDIS_HOST}")

def insert_face(path, gender, beard, glasses, approach, timestamp=None):
    """Insert face path and metadata into local redis instance"""
    face_id = os.path.basename(path)

    key = f"face:{face_id}"
    if timestamp is None:
        timestamp = time.time()

    mapping = {
        "path": path,
        "gender": gender,
        "beard": beard,
        "glasses": glasses,
        "approach": approach,
        "timestamp": timestamp
    }
    r.hset(key, mapping=mapping)

    # Index Sets for Filtering
    r.sadd(f"idx:gender:{gender}", face_id)
    r.sadd(f"idx:beard:{beard}", face_id)
    r.sadd(f"idx:approach:{approach}", face_id)
    r.sadd(f"idx:glasses:{glasses}", face_id)

    # Timeline (Sorted Set) for chronological order
    r.zadd("idx:timeline", {face_id: timestamp})

def delete_face(face_id):
    """Remove a face and its indices from Redis and delets the .jpg from filesystem."""
    key = f"face:{face_id}"
    data = r.hgetall(key)

    if not data:
        return

    gender = data.get("gender")
    beard = data.get("beard")
    glasses = data.get("glasses")
    approach = data.get("approach")

    # Remove from indices
    if gender is not None: r.srem(f"idx:gender:{gender}", face_id)
    if beard is not None: r.srem(f"idx:beard:{beard}", face_id)
    if glasses is not None: r.srem(f"idx:glasses:{glasses}", face_id)
    if approach is not None: r.srem(f"idx:approach:{approach}", face_id)

    # Remove from timeline
    r.zrem("idx:timeline", face_id)

    # Remove the hash
    r.delete(key)

    if os.path.exists(f"{SAVE_DIR}/{face_id}"):
        os.remove(f"{SAVE_DIR}/{face_id}")

    return True

def get_all_face_ids():
    """Retrieve all face IDs (filenames) currently indexed."""
    return r.zrange("idx:timeline", 0, -1)

def get_filtered_faces(gender=None, beard=None, glasses=None, approach=None, n_to_return=20):
    sets_to_intersect = []

    if gender is not None and gender != -1:
        sets_to_intersect.append(f"idx:gender:{gender}")
    if beard is not None and beard != -1:
        sets_to_intersect.append(f"idx:beard:{beard}")
    if glasses is not None and glasses != -1:
        sets_to_intersect.append(f"idx:glasses:{glasses}")
    if approach is not None and approach != -1:
        sets_to_intersect.append(f"idx:approach:{approach}")


    final_ids = []

    if not sets_to_intersect:
        # Get all IDs from timeline, reversed (newest first)
        final_ids = r.zrevrange("idx:timeline", 0, -1)
    else:
        # Intersect the filter sets to find matches
        matches = r.sinter(sets_to_intersect)
        final_ids = sorted(list(matches), key=lambda x: r.zscore("idx:timeline", x) or 0, reverse=True)

    # Take the n_to_return latest generated
    final_ids = final_ids[:n_to_return]
    results = []
    for fid in final_ids:
        data = r.hgetall(f"face:{fid}")
        if data:
            results.append({
                "id": fid,
                "path": data.get("path"),
                "gender": int(data.get("gender")),
                "beard": int(data.get("beard")),
                "glasses": int(data.get("glasses")),
                "approach": data.get("approach"),
                "timestamp": float(data.get("timestamp"))
            })

    return results





def sync_filesystem_with_db():
    """Scans FACES folder and syncs state with Redis."""
    if not os.path.exists(SAVE_DIR):
        return

    print("Syncing FACES folder with Redis...")

    on_disk = set([f for f in os.listdir(SAVE_DIR) if f.endswith(".jpg")])

    in_db = set(get_all_face_ids())

    to_add = on_disk - in_db
    for filename in to_add:
        try:
            name_parts = filename.replace(".jpg", "").split("_")
            if len(name_parts) != 3:
                continue

            _, class_code, approach = name_parts

            if len(class_code) != 3:
                 continue

            gender = int(class_code[0])
            glasses = int(class_code[1])
            beard = int(class_code[2])

            full_path = f"/{SAVE_DIR}/{filename}"
            file_ts = os.path.getmtime(os.path.join(SAVE_DIR, filename))

            insert_face(full_path, gender, beard, glasses, approach, timestamp=file_ts)
            print(f"Synced: Added {filename}")

        except Exception as e:
            print(f"Error syncing file {filename}: {e}")

    to_remove = in_db - on_disk
    for filename in to_remove:
        delete_face(filename)
        print(f"Synced: Removed {filename}")

    print("Sync complete.")



if __name__ == '__main__':
    init_db()
