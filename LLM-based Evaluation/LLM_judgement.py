import os
import re
import json
import time
import argparse
from openai import OpenAI


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_caption(data, sample_id):
    caption = data[sample_id]
    if isinstance(caption, list):
        caption = caption[0]
    return str(caption)


def parse_judgement(text):
    match = re.search(
        r"\b(Yes|No)\s*[,，]\s*(Yes|No)\b",
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1).capitalize(), match.group(2).capitalize()

    return None, None


def main(args):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    candidate_data = load_json(args.candidate)
    reference_data = load_json(args.reference)

    with open(args.prompt_fp, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    os.makedirs(args.save_dir, exist_ok=True)

    sample_ids = sorted(
        set(candidate_data.keys()) & set(reference_data.keys())
    )

    all_results = {}

    for idx, sample_id in enumerate(sample_ids):
        candidate = get_caption(candidate_data, sample_id)
        reference = get_caption(reference_data, sample_id)

        prompt = (
            prompt_template
            .replace("{{candidate}}", candidate)
            .replace("{{reference}}", reference)
        )

        print(f"Evaluating {sample_id} ({idx + 1}/{len(sample_ids)})")

        raw_outputs = []
        parsed_results = []

        for attempt in range(args.retry):
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=args.temperature,
                    max_tokens=20,
                    n=args.n
                )

                for choice in response.choices:
                    output = choice.message.content.strip()
                    coverage, substitutability = parse_judgement(output)

                    raw_outputs.append(output)
                    parsed_results.append({
                        "coverage": coverage,
                        "substitutability": substitutability
                    })

                break

            except Exception as e:
                print(f"Error on {sample_id}: {e}")
                if attempt < args.retry - 1:
                    time.sleep(2)

        all_results[sample_id] = {
            "candidate": candidate,
            "reference": reference,
            "raw_outputs": raw_outputs,
            "parsed_results": parsed_results
        }

        time.sleep(args.sleep)


    raw_path = os.path.join(args.save_dir, "judgement_results.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)


    statistics = {}

    for metric in ["coverage", "substitutability"]:
        yes_count = 0
        no_count = 0

        for result in all_results.values():
            for item in result["parsed_results"]:
                value = item[metric]

                if value == "Yes":
                    yes_count += 1
                elif value == "No":
                    no_count += 1

        valid_count = yes_count + no_count

        statistics[metric] = {
            "yes_count": yes_count,
            "no_count": no_count,
            "valid_count": valid_count,
            "yes_percentage": (
                round(yes_count / valid_count * 100, 2)
                if valid_count > 0 else 0
            ),
            "no_percentage": (
                round(no_count / valid_count * 100, 2)
                if valid_count > 0 else 0
            )
        }

    stats_path = os.path.join(args.save_dir, "judgement_statistics.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(statistics, f, ensure_ascii=False, indent=4)

    print("\nEvaluation completed.")
    print(json.dumps(statistics, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--candidate", type=str, required=True)
    parser.add_argument("--reference", type=str, required=True)
    parser.add_argument("--prompt_fp", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="results/judgement")

    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--retry", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=0.5)

    args = parser.parse_args()
    main(args)