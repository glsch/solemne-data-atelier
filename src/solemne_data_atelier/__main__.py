import argparse
import logging
import sys

from solemne_data_atelier import setup_logging, __version__, __package_name__

logger = logging.getLogger(__name__)

def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="solemne_data_atelier",
        description="A text reuse hackathon toolkit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # todo: add examples
        epilog="""
    
    Examples:
        """,
    )

    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress non-error output.",
    )
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        metavar="<command>",
    )

    from solemne_data_atelier.commands import (
        add_download_parser,
        add_preprocess_parser,
        add_vectorstore_parser,
    )

    add_download_parser(subparsers)
    add_preprocess_parser(subparsers)
    add_vectorstore_parser(subparsers)

    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    if args.quiet:
        log_level = logging.ERROR
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    setup_logging()
    logging.getLogger().setLevel(log_level)

    logger.debug(f"Running {__package_name__} v{__version__}")

    if args.command is None:
        parser.print_help()
        return 0

    if hasattr(args, "func"):
        return args.func(args)

    parser.print_help()
    return 0


def cli() -> None:
    sys.exit(main())


if __name__ == "__main__":
    cli()
