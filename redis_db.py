import redis
import os
import time

REDIS_HOST = "localhost"
REDIS_PORT = 6379
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def init_db():
    """Checks connection."""
    try:
        r.ping()
        print(f"Connected to Redis at {REDIS_HOST}")
    except redis.ConnectionError:
        print(f"Error: Could not connect to Redis at {REDIS_HOST}")

def insert_face(path, gender, beard, glasses, approach):
    """Insert face path and metadata into local redis instance"""
    face_id = os.path.basename(path)

    key = f"face:{face_id}"
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
        # SINTER returns IDs present in ALL provided sets
        matches = r.sinter(sets_to_intersect)
        # Sort them by checking their score in the timeline ZSET
        # Should be changed
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


if __name__ == '__main__':
    init_db()
