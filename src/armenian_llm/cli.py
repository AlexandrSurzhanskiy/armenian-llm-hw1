"""Единые команды для запуска этапов из терминала и Jupyter."""

import argparse
import runpy
import sys


COMMANDS = {
    "download-armweb": "data.armweb.download",
    "inspect-armweb": "data.armweb.inspect",
    "analyze-armweb": "data.armweb.analyze",
    "inspect-repetition": "data.armweb.repetition",
    "prepare-armweb": "data.armweb.prepare",
    "sample-tokenizer": "tokenization.sample",
    "train-tokenizer": "tokenization.train",
    "encode": "data.encode",
    "download-wikipedia": "data.wikipedia.download",
    "inspect-wikipedia": "data.wikipedia.inspect",
    "clean-wikipedia": "data.wikipedia.clean",
    "filter-wikipedia": "data.wikipedia.filter",
    "prepare-wikipedia": "data.wikipedia.prepare",
    "mixture": "data.mixture",
    "pretrain": "training.pretrain",
    "continue-pretrain": "training.continue_pretrain",
    "sft": "training.sft",
    "benchmark": "experiments.benchmark",
    "check-model": "experiments.check_model",
    "pilot-222m": "experiments.pilot_222m",
    "evaluate": "evaluation.heldout",
    "evaluate-final": "evaluation.final",
    "evaluate-facts": "evaluation.facts",
    "evaluate-sft": "evaluation.sft",
    "compare-generations": "evaluation.generations",
    "report": "reporting.build",
    "export-reports": "reporting.export_reports",
    "export-model": "reporting.export_model",
}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="armenian-llm", description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    if not arguments or arguments[0] in ("-h", "--help"):
        parser.print_help()
        return
    parsed = parser.parse_args(arguments[:1])
    module = "armenian_llm." + COMMANDS[parsed.command]
    original = sys.argv
    try:
        sys.argv = [module, *arguments[1:]]
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    finally:
        sys.argv = original
