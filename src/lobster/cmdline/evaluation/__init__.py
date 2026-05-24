"""Standalone evaluation CLIs for LeFlur outputs.

Each module under :mod:`lobster.cmdline.evaluation` exposes an
``argparse``-based main entry point that wraps the corresponding evaluator in
:mod:`lobster.metrics.protein_ligand`.  These scripts are invoked directly via
``python -m lobster.cmdline.evaluation.<module>`` and are not re-exported here
to keep the import surface small.
"""
