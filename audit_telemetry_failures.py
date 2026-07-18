import json
import click


@click.command()
@click.option('--telemetry_file', '-t', required=True, help="Path to the vllm-telemetry log file.")
def main(telemetry_file: str):
    print("=== Analyzing Telemetry Parse Failures ===")
    failure_count = 0

    with open(telemetry_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            log = json.loads(line)

            if log.get("parsed_query_reduction") is None:
                failure_count += 1
                query_id = log.get("query_id")
                raw_text = log.get("generated_text", "").strip()

                print(f"\n[Failure #{failure_count}] Query ID: {query_id}")
                print(f"Raw Model Response: {raw_text}")
                print("-" * 50)

                if failure_count >= 10:
                    print("\n... truncated. First 10 failures displayed above.")
                    break


if __name__ == '__main__':
    main()