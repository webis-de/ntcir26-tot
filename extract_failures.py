import json
import click


@click.command()
@click.option('--input_file', '-i', required=True, help="Path to your clean output JSONL file.")
def main(input_file: str):
    failed_ids = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("query_reduction") is None:
                failed_ids.append(record["query_id"])

    print(f"Total failed records found: {len(failed_ids)}")
    with open("failed_query_ids.txt", "w") as out:
        out.write("\n".join(failed_ids))

    print("Saved failed IDs to failed_query_ids.txt")


if __name__ == '__main__':
    main()