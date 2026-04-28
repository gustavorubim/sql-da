import json
import sys
from pathlib import Path

import pytest

from sql_da import _sitecustomize
from sql_da.config import DEFAULT_MODEL, ProjectConfig, config_to_dict, load_env_file
from sql_da.lineage import LineageGraph

ROOT = Path(__file__).resolve().parents[1]


def test_config_resolves_paths_and_defaults(monkeypatch):
    monkeypatch.delenv("SQL_DA_MODEL", raising=False)
    config = ProjectConfig.from_yaml(ROOT / "configs/demo.yaml")

    assert config.database.path == ROOT / "data/demo_finance.db"
    assert config.database.uri == f"sqlite:///{ROOT / 'data/demo_finance.db'}"
    assert config.lineage_path == ROOT / "configs/lineage.yaml"
    assert config.model.name == DEFAULT_MODEL
    assert config.model.args == {}
    assert config.agent.memory_paths == (ROOT / "memory/AGENTS.md",)
    assert config.agent.skill_paths == (ROOT / "skills",)
    assert config.analysis.materiality_abs == 5000.0
    config_dict = config_to_dict(config)
    assert "revenue" in config_dict["metrics"]
    assert str(ROOT / "skills") in config_dict["agent_skill_paths"]
    assert config_dict["model_temperature"] == 0.0
    assert config_dict["model_args"] == {}


def test_config_honors_model_env_override(monkeypatch):
    monkeypatch.setenv("SQL_DA_MODEL", "google_genai:custom")

    config = ProjectConfig.from_yaml(ROOT / "configs/demo.yaml")

    assert config.model.name == "google_genai:custom"


def test_config_loads_chat_google_genai_args(tmp_path, monkeypatch):
    monkeypatch.delenv("SQL_DA_MODEL", raising=False)
    config_path = tmp_path / "demo.yaml"
    config_path.write_text(
        """
model:
  name: google_genai:gemini-3-flash-preview
  temperature: 0.1
  args:
    top_p: 0.9
    top_k: 32
    max_tokens: 2048
    request_timeout: 90
database:
  path: demo.db
lineage_path: lineage.yaml
report_dir: reports
""",
        encoding="utf-8",
    )

    config = ProjectConfig.from_yaml(config_path)

    assert config.model.temperature == 0.1
    assert config.model.args == {
        "top_p": 0.9,
        "top_k": 32,
        "max_tokens": 2048,
        "request_timeout": 90,
    }
    assert config_to_dict(config)["model_args"]["max_tokens"] == 2048


def test_config_rejects_non_mapping_model_args(tmp_path):
    config_path = tmp_path / "demo.yaml"
    config_path.write_text(
        """
model:
  args:
    - top_p
database:
  path: demo.db
lineage_path: lineage.yaml
report_dir: reports
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"model\.args must be a mapping"):
        ProjectConfig.from_yaml(config_path)


def test_load_env_file_does_not_override_existing_env(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "EXISTING=from-file\nNEW_VALUE=created\n# ignored\nBADLINE\n", encoding="utf-8"
    )
    monkeypatch.setenv("EXISTING", "already")

    load_env_file(env_path)

    assert __import__("os").environ["EXISTING"] == "already"
    assert __import__("os").environ["NEW_VALUE"] == "created"
    load_env_file(tmp_path / "missing.env")


def test_config_supports_absolute_paths_outside_configs(tmp_path, monkeypatch):
    monkeypatch.delenv("SQL_DA_MODEL", raising=False)
    db_path = tmp_path / "finance.db"
    lineage_path = ROOT / "configs/lineage.yaml"
    report_dir = tmp_path / "reports"
    config_path = tmp_path / "demo.yaml"
    config_path.write_text(
        f"""
database:
  uri: sqlite:///relative.db
  path: {db_path}
lineage_path: {lineage_path}
report_dir: {report_dir}
agent:
  memory_paths:
    - {tmp_path / "AGENTS.md"}
  skill_paths:
    - {tmp_path / "skills"}
reward:
  output_path: {tmp_path / "reward.json"}
""",
        encoding="utf-8",
    )

    config = ProjectConfig.from_yaml(config_path)

    assert config.project_root == tmp_path
    assert config.database.path == db_path
    assert config.lineage_path == lineage_path
    assert config.report_dir == report_dir
    assert config.agent.memory_paths == (tmp_path / "AGENTS.md",)


def test_sitecustomize_adds_local_src_from_direct_url(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    package_dir = project_root / "src" / "sql_da"
    package_dir.mkdir(parents=True)
    site_dir = tmp_path / "site-packages"
    dist_info = site_dir / "sql_da-0.1.0.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "direct_url.json").write_text(
        json.dumps({"url": project_root.as_uri()}), encoding="utf-8"
    )
    monkeypatch.setattr(_sitecustomize, "__file__", str(site_dir / "sitecustomize.py"))
    sys.path = [path for path in sys.path if path != str(project_root / "src")]

    _sitecustomize._add_local_src_path()

    assert str(project_root / "src") in sys.path


def test_lineage_validates_and_traces_metrics():
    graph = LineageGraph.from_yaml(ROOT / "configs/lineage.yaml")

    assert graph.validate() == []
    assert graph.trace_metric("net_income")[0] == "mart_financials_monthly"
    assert graph.trace_metric("Net Income")[0] == "mart_financials_monthly"
    assert "invoices" in graph.trace_metric("revenue")
    assert graph.trace_upstream("mart_financials_monthly", max_depth=1) == [
        "mart_financials_monthly",
        "mart_revenue_monthly",
        "mart_cogs_monthly",
        "mart_payroll_monthly",
        "mart_vendor_spend_monthly",
    ]
    assert any(node["node_id"] == "gl_entries" for node in graph.list_nodes())
    relationships = graph.list_relationship_specs()
    product_relationships = graph.list_relationship_specs("products", limit=1)
    transformations = graph.list_transformation_specs()
    product_transformations = graph.list_transformation_specs("products")
    semantic_specs = graph.semantic_specs()
    semantic_summary = graph.semantic_summary()
    search_results = graph.search_semantic_specs("cogs rate product", limit=2)
    product_context = graph.get_node_context("products", limit=2)

    assert any(
        relationship["relationship_id"] == "invoice_product" for relationship in relationships
    )
    assert product_relationships == [
        next(
            relationship
            for relationship in relationships
            if relationship["relationship_id"] == "invoice_product"
        )
    ]
    assert any(
        transformation["transformation_id"] == "cloud_cogs_bill_amount"
        and "products.cogs_rate" in transformation["inputs"]
        for transformation in transformations
    )
    assert [transformation["transformation_id"] for transformation in product_transformations] == [
        "cloud_cogs_bill_amount"
    ]
    assert semantic_specs["relationships"] == relationships
    assert semantic_specs["transformations"] == transformations
    assert semantic_summary["relationship_count"] == len(relationships)
    assert semantic_summary["transformation_count"] == len(transformations)
    assert "retrieve" in semantic_summary["retrieval_guidance"]
    assert search_results[0]["id"] == "cloud_cogs_bill_amount"
    assert product_context["node"]["node_id"] == "products"
    assert len(product_context["relationships"]) == 2
    assert product_context["transformations"][0]["transformation_id"] == "cloud_cogs_bill_amount"


def test_lineage_reports_bad_references(tmp_path):
    lineage_path = tmp_path / "lineage.yaml"
    lineage_path.write_text(
        """
nodes:
  broken:
    columns: []
    upstream: [missing]
metric_mappings:
  revenue:
    node: missing
""",
        encoding="utf-8",
    )

    graph = LineageGraph.from_yaml(lineage_path)

    errors = graph.validate()
    assert "broken is missing type" in errors
    assert "broken has no columns" in errors
    assert "broken references missing upstream node missing" in errors
    assert "metric revenue references missing node missing" in errors
    assert "metric revenue is missing value_column" in errors
    with pytest.raises(KeyError):
        graph.trace_upstream("missing")
    with pytest.raises(KeyError):
        graph.trace_metric("missing_metric")


def test_lineage_reports_bad_relationship_and_transformation_specs(tmp_path):
    lineage_path = tmp_path / "lineage.yaml"
    lineage_path.write_text(
        """
nodes:
  invoices:
    type: raw_table
    grain: one row
    columns: [invoice_id, amount]
  products:
    type: dimension
    grain: one row
    columns: [product_id]
relationships:
  bad_join:
    left: invoices.missing_column
    right: missing_node.product_id
transformations:
  bad_transform:
    output: products.missing_output
    expression: SUM(invoices.amount)
    source_nodes: [missing_node]
    inputs:
      - invoices.amount
      - missing_node.amount
""",
        encoding="utf-8",
    )

    errors = LineageGraph.from_yaml(lineage_path).validate()

    assert "relationship bad_join left invoices.missing_column references missing column" in (
        "; ".join(errors)
    )
    assert "relationship bad_join right missing_node.product_id does not match a known node" in (
        "; ".join(errors)
    )
    assert (
        "transformation bad_transform output products.missing_output references missing column"
        in ("; ".join(errors))
    )
    assert "transformation bad_transform references missing source node missing_node" in errors
    assert (
        "transformation bad_transform reference missing_node.amount does not match a known node"
        in ("; ".join(errors))
    )


def test_lineage_rejects_malformed_spec_sections(tmp_path):
    lineage_path = tmp_path / "lineage.yaml"
    lineage_path.write_text(
        """
nodes: {}
relationships: not-a-mapping
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="relationships must be a mapping or list"):
        LineageGraph.from_yaml(lineage_path)

    lineage_path.write_text(
        """
nodes: {}
transformations:
  - output: x.y
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"transformations\[1\] is missing id"):
        LineageGraph.from_yaml(lineage_path)
