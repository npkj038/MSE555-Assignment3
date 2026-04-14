import json
import csv

def json_to_long_csv(json_file, csv_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for item in data:
        client_id = item.get("client_id")
        session = item.get("note_number")  # using note_number as session
        vector = item.get("estimated_trajectory_vector", [])

        if not isinstance(vector, list):
            continue

        # expand vector into multiple rows
        for i, score in enumerate(vector, start=1):
            rows.append({
                "client_id": client_id,
                "session": f"{session}_{i}",  # or just session if you prefer
                "score": score
            })

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["client_id", "session", "score"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved to {csv_file}")

# run
json_to_long_csv("output\\q1\\scored_notes.json", "output\\q1\\scored_notes.csv")