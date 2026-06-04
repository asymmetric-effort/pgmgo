"""
Fixture generator for readwrite package.

Tests BIF and XMLBIF round-trip: write student BN, read back, verify
node/edge counts and structure match.
"""

import warnings
warnings.filterwarnings("ignore")

import tempfile
import os
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.readwrite import BIFWriter, BIFReader, XMLBIFWriter, XMLBIFReader


def generate() -> list[dict]:
    """Generate readwrite test cases."""
    test_cases = []

    test_cases.append(_test_bif_roundtrip())
    test_cases.append(_test_xmlbif_roundtrip())

    return test_cases


def _build_student_bn():
    """Build the classic student BN with state names for serialization."""
    edges = [("D", "G"), ("I", "G"), ("G", "L"), ("I", "S")]
    bn = BayesianNetwork(edges)

    sn = {
        "D": ["d0", "d1"],
        "I": ["i0", "i1"],
        "G": ["g0", "g1", "g2"],
        "L": ["l0", "l1"],
        "S": ["s0", "s1"],
    }

    cpd_d = TabularCPD("D", 2, [[0.6], [0.4]], state_names={"D": sn["D"]})
    cpd_i = TabularCPD("I", 2, [[0.7], [0.3]], state_names={"I": sn["I"]})
    cpd_g = TabularCPD(
        "G", 3,
        [[0.3, 0.05, 0.9, 0.5],
         [0.4, 0.25, 0.08, 0.3],
         [0.3, 0.7, 0.02, 0.2]],
        evidence=["D", "I"], evidence_card=[2, 2],
        state_names={"G": sn["G"], "D": sn["D"], "I": sn["I"]},
    )
    cpd_l = TabularCPD(
        "L", 2,
        [[0.1, 0.4, 0.99],
         [0.9, 0.6, 0.01]],
        evidence=["G"], evidence_card=[3],
        state_names={"L": sn["L"], "G": sn["G"]},
    )
    cpd_s = TabularCPD(
        "S", 2,
        [[0.95, 0.2],
         [0.05, 0.8]],
        evidence=["I"], evidence_card=[2],
        state_names={"S": sn["S"], "I": sn["I"]},
    )

    bn.add_cpds(cpd_d, cpd_i, cpd_g, cpd_l, cpd_s)
    return bn


def _test_bif_roundtrip():
    """Write student BN to BIF, read back, verify structure."""
    bn = _build_student_bn()

    with tempfile.NamedTemporaryFile(suffix=".bif", delete=False, mode="w") as f:
        bif_path = f.name

    try:
        writer = BIFWriter(bn)
        writer.write_bif(bif_path)

        reader = BIFReader(bif_path)
        bn2 = reader.get_model()

        # Read the BIF content for the fixture
        with open(bif_path, "r") as f:
            bif_content = f.read()
    finally:
        os.unlink(bif_path)

    return {
        "name": "bif_roundtrip",
        "description": "Write student BN to BIF, read back, verify structure",
        "input": {
            "bif_content": bif_content,
            "original_nodes": sorted(bn.nodes()),
            "original_edges": sorted([list(e) for e in bn.edges()]),
        },
        "expected": {
            "nodes": sorted(bn2.nodes()),
            "edges": sorted([list(e) for e in bn2.edges()]),
            "num_nodes": len(bn2.nodes()),
            "num_edges": len(bn2.edges()),
            "num_cpds": len(bn2.cpds),
        },
    }


def _test_xmlbif_roundtrip():
    """Write student BN to XMLBIF, read back, verify structure."""
    bn = _build_student_bn()

    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w") as f:
        xml_path = f.name

    try:
        writer = XMLBIFWriter(bn)
        writer.write_xmlbif(xml_path)

        reader = XMLBIFReader(xml_path)
        bn2 = reader.get_model()

        with open(xml_path, "r") as f:
            xmlbif_content = f.read()
    finally:
        os.unlink(xml_path)

    return {
        "name": "xmlbif_roundtrip",
        "description": "Write student BN to XMLBIF, read back, verify structure",
        "input": {
            "xmlbif_content": xmlbif_content,
            "original_nodes": sorted(bn.nodes()),
            "original_edges": sorted([list(e) for e in bn.edges()]),
        },
        "expected": {
            "nodes": sorted(bn2.nodes()),
            "edges": sorted([list(e) for e in bn2.edges()]),
            "num_nodes": len(bn2.nodes()),
            "num_edges": len(bn2.edges()),
            "num_cpds": len(bn2.cpds),
        },
    }
