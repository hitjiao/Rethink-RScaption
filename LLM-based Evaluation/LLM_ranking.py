
import os
import re
import json
import time
import argparse
from pathlib import Path
from collections import Counter

from openai import OpenAI


CANDIDATE_NAMES = [
    "Candidate1",
    "Candidate2",
    "Candidate3",
    "Candidate4",
    "Candidate5"
]

METRICS = ["accuracy", "fineness", "fluency", "richness"]

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_caption(data, key):
    value = data[key]

    if isinstance(value, list):
        if len(value) == 0:
            raise ValueError(f"Empty caption for sample {key}")
        return str(value[0])
    return str(value)

def extract_ranking(text):
    if not text:
        return None

    matches = re.findall(r"(?<!\d)[1-5]{5}(?!\d)", text)

    for ranking in matches:
        if sorted(ranking) == ["1", "2", "3", "4", "5"]:
            return ranking

    numbers = re.findall(r"\d", text)

    if len(numbers) == 5:
        ranking = "".join(numbers)
        if sorted(ranking) == ["1", "2", "3", "4", "5"]:
            return ranking
    return None

def build_prompt(prompt_template, candidates, reference=None):
    prompt = prompt_template

    for i, candidate in enumerate(candidates, start=1):
        prompt = prompt.replace(
            f"{{{{candidate{i}}}}}",
            candidate
        )

    if "{{reference}}" in prompt:
        if reference is None:
            raise ValueError(
                "The accuracy prompt requires {{reference}}, "
                "but no reference caption was provided."
            )

        prompt = prompt.replace("{{reference}}", reference)
    return prompt

def evaluate(
    client,
    model,
    prompt,
    n=10,
    max_retries=5
):

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=10,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                n=n
            )

            rankings = []

            for choice in response.choices:
                content = choice.message.content or ""
                ranking = extract_ranking(content)

                rankings.append({
                    "raw_output": content,
                    "ranking": ranking
                })

            while len(rankings) < n:
                rankings.append({
                    "raw_output": "",
                    "ranking": None
                })

            return rankings

        except Exception as e:
            print(f"API error (attempt {attempt + 1}/{max_retries}): {e}")

            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))

    return [
        {
            "raw_output": "",
            "ranking": None
        }
        for _ in range(n)
    ]


def calculate_statistics(results):
    statistics = {}

    for i, candidate_name in enumerate(CANDIDATE_NAMES):

        all_ranks = []

        for sample_id, responses in results.items():
            for response in responses:
                ranking = response.get("ranking")

                if ranking and len(ranking) == 5:
                    all_ranks.append(int(ranking[i]))

        valid_count = len(all_ranks)

        if valid_count == 0:
            statistics[candidate_name] = {
                "valid_count": 0,
                "mean_rank": None,
                "rank_distribution": {
                    str(rank): 0 for rank in range(1, 6)
                },
                "rank_1_count": 0,
                "rank_1_percentage": 0
            }
            continue

        rank_distribution = Counter(all_ranks)

        statistics[candidate_name] = {
            "valid_count": valid_count,

            "mean_rank": round(
                sum(all_ranks) / valid_count, 4
            ),
            "rank_distribution": {
                str(rank): rank_distribution.get(rank, 0)
                for rank in range(1, 6)
            },
            "rank_1_count": rank_distribution.get(1, 0),
            "rank_1_percentage": round(
                rank_distribution.get(1, 0) / valid_count * 100,
                2
            )
        }

    return statistics


def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--candidate1", type=str, required=True)
    parser.add_argument("--candidate2", type=str, required=True)
    parser.add_argument("--candidate3", type=str, required=True)
    parser.add_argument("--candidate4", type=str, required=True)
    parser.add_argument("--candidate5", type=str, required=True)

    parser.add_argument("--reference", type=str, required=True)

    parser.add_argument("--accuracy_prompt", type=str, required=True)
    parser.add_argument("--fineness_prompt", type=str, required=True)
    parser.add_argument("--fluency_prompt", type=str, required=True)
    parser.add_argument("--richness_prompt", type=str, required=True)

    parser.add_argument("--save_dir", type=str, default="./results")

    parser.add_argument(
        "--key",
        type=str,
        default=os.getenv("OPENAI_API_KEY")
    )
    parser.add_argument(
        "--base",
        type=str,
        default=os.getenv("OPENAI_BASE_URL", "")
    )
    parser.add_argument("--model", type=str, default="gpt-4o")

    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    if not args.key:
        raise ValueError(
            "Please provide your API key using --key "
            "or OPENAI_API_KEY."
        )

    client = OpenAI(
        api_key=args.key,
        base_url=args.base if args.base else None
    )

    os.makedirs(args.save_dir, exist_ok=True)

    candidate_files = [
        args.candidate1,
        args.candidate2,
        args.candidate3,
        args.candidate4,
        args.candidate5
    ]

    candidates_data = [
        load_json(path) for path in candidate_files
    ]

    reference_data = load_json(args.reference)

    sample_ids = list(candidates_data[0].keys())

    for i, data in enumerate(candidates_data[1:], start=2):
        missing = set(sample_ids) - set(data.keys())

        if missing:
            raise ValueError(
                f"Candidate{i} is missing sample IDs: "
                f"{list(missing)[:10]}"
            )

    missing_reference = set(sample_ids) - set(reference_data.keys())

    if missing_reference:
        raise ValueError(
            "Reference file is missing sample IDs: "
            f"{list(missing_reference)[:10]}"
        )

    sample_ids = [
        key for key in sample_ids
        if key.isdigit() and int(key) < 10000
    ]

    if args.limit is not None:
        sample_ids = sample_ids[:args.limit]

    print(f"Total samples: {len(sample_ids)}")
    print(f"Model: {args.model}")
    print(f"Repeats per metric: {args.n}")

    prompt_paths = {
        "accuracy": args.accuracy_prompt,
        "fineness": args.fineness_prompt,
        "fluency": args.fluency_prompt,
        "richness": args.richness_prompt
    }

    prompts = {}

    for metric, path in prompt_paths.items():
        with open(path, "r", encoding="utf-8") as f:
            prompts[metric] = f.read()

    all_results = {}

    for metric in METRICS:

        print("\n" + "=" * 60)
        print(f"Evaluating metric: {metric.upper()}")
        print("=" * 60)
        metric_results = {}
        for sample_index, key in enumerate(sample_ids, start=1):
            try:
                candidates = [
                    get_caption(data, key)
                    for data in candidates_data
                ]
                reference = get_caption(reference_data, key)
                cur_prompt = build_prompt(
                    prompts[metric],
                    candidates,
                    reference
                )
                responses = evaluate(
                    client=client,
                    model=args.model,
                    prompt=cur_prompt,
                    n=args.n
                )
                metric_results[key] = responses

                valid_count = sum(
                    1 for item in responses
                    if item["ranking"] is not None
                )
                print(
                    f"[{metric.upper()}] "
                    f"{sample_index}/{len(sample_ids)} "
                    f"ID: {key} "
                    f"Valid: {valid_count}/{args.n}"
                )
                time.sleep(args.sleep)
            except Exception as e:
                print(f"Sample {key} failed: {e}")

                metric_results[key] = [
                    {
                        "raw_output": "",
                        "ranking": None
                    }
                    for _ in range(args.n)
                ]

        raw_path = os.path.join(
            args.save_dir,
            f"{metric}_raw.json"
        )

        save_json(metric_results, raw_path)

        metric_statistics = calculate_statistics(
            metric_results
        )

        stats_path = os.path.join(
            args.save_dir,
            f"{metric}_statistics.json"
        )

        save_json(metric_statistics, stats_path)

        all_results[metric] = {
            "raw_results": metric_results,
            "statistics": metric_statistics
        }

        print(f"\n{metric.upper()} statistics:")

        for candidate, stats in metric_statistics.items():
            print(
                f"{candidate}: "
                f"Mean Rank = {stats['mean_rank']}, "
                f"Rank 1 = {stats['rank_1_count']}, "
                f"Valid = {stats['valid_count']}"
            )

    summary = {
        metric: all_results[metric]["statistics"]
        for metric in METRICS
    }

    save_json(
        summary,
        os.path.join(args.save_dir, "all_metrics_statistics.json")
    )

    print("\n" + "=" * 60)
    print("All evaluations completed!")
    print(f"Results saved in: {args.save_dir}")
    print("=" * 60)
