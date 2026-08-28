import argparse


def build_greet_parser() -> argparse.ArgumentParser:
    """Build the argument parser for a small greeting CLI.

    Returns:
        An ArgumentParser with --name (required) and --lang
        (optional, default "en") arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        required=True,
        help="The name of the person to greet."
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="The language of the person to greet."
    )
    return parser


GREETINGS = {
    "en": "Hello, {name}!",
    "tr": "Merhaba, {name}!",
}


def main():
    parser = build_greet_parser()
    args = parser.parse_args()
    template = GREETINGS.get(args.lang, GREETINGS["en"])
    print(template.format(name=args.name))


if __name__ == "__main__":
    main()