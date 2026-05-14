import json
from pathlib import Path

from app.config import CLASS_NAMES
from app.inference import predict


EVAL_DIR = Path("data/eval")
REPORT_PATH = Path("reports/evaluation_results.json")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
def collect_images():
    samples = []

    for label in CLASS_NAMES:
        class_dir = EVAL_DIR / label

        if not class_dir.exists():
            continue

        for path in class_dir.iterdir():
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, label))

    return samples


def main():
    samples = collect_images()

    if not samples:
        print("No evaluation images found.")
        print("Add labeled images under data/eval/{heart,long,oval,round,square}/")
        return

    total = 0
    correct = 0
    confidence_sum = 0.0
    low_confidence_count = 0

    per_class = {
        label: {"total": 0, "correct": 0}
        for label in CLASS_NAMES
    }

    confusion_matrix = {
        true_label: {pred_label: 0 for pred_label in CLASS_NAMES}
        for true_label in CLASS_NAMES
    }

    failures = []

    for image_path, true_label in samples:
        image_bytes = image_path.read_bytes()
        result = predict(image_bytes)

        total += 1
        per_class[true_label]["total"] += 1

        if "error" in result:
            failures.append({
                "image": str(image_path),
                "true_label": true_label,
                "error": result["error"],
            })
            continue

        pred_label = result["face_type"]
        confidence = result["confidence"]

        confidence_sum += confidence

        if result["confidence_label"] == "low":
            low_confidence_count += 1

        confusion_matrix[true_label][pred_label] += 1

        if pred_label == true_label:
            correct += 1
            per_class[true_label]["correct"] += 1
        else:
            failures.append({
                "image": str(image_path),
                "true_label": true_label,
                "predicted_label": pred_label,
                "confidence": confidence,
            })

    per_class_accuracy = {}

    for label, stats in per_class.items():
        if stats["total"] == 0:
            per_class_accuracy[label] = None
        else:
            per_class_accuracy[label] = round(stats["correct"] / stats["total"], 4)

    results = {
        "total_images": total,
        "correct_predictions": correct,
        "accuracy": round(correct / total, 4) if total else None,
        "average_confidence": round(confidence_sum / total, 6) if total else None,
        "low_confidence_count": low_confidence_count,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion_matrix,
        "failures": failures[:25],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))
    print(f"Saved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()