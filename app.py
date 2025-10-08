#!pip install -q pandas seaborn matplotlib pyyaml gradio google-generativeai

# ============================================================
# 📚 Imports
# ============================================================
import io
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
import gradio as gr
import google.generativeai as genai

# ============================================================
# 🎨 Global style
# ============================================================
sns.set(style="whitegrid")
CORAL = "#FF6F61"

# ============================================================
# 🧩 Helpers: Data loading and agents parsing
# ============================================================
def detect_format_and_load(text_or_bytes, filename=None):
    """
    Load dataset from CSV or JSON (file or pasted text).
    Returns: (df, msg)
    """
    if text_or_bytes is None:
        return None, "No dataset provided."

    # If it's a file-like object (bytes), try using filename hint
    if isinstance(text_or_bytes, bytes):
        if filename and filename.lower().endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(text_or_bytes))
                return df, f"Loaded CSV: {filename}"
            except Exception as e:
                return None, f"Error reading CSV: {e}"
        else:
            # Try JSON fallback
            try:
                obj = json.load(io.BytesIO(text_or_bytes))
                df = pd.json_normalize(obj)
                return df, f"Loaded JSON: {filename or 'uploaded file'}"
            except Exception as e:
                return None, f"Error reading JSON: {e}"

    # If it's pasted text (string)
    if isinstance(text_or_bytes, str):
        txt = text_or_bytes.strip()
        # Try JSON first
        try:
            obj = json.loads(txt)
            df = pd.json_normalize(obj)
            return df, "Loaded JSON from pasted text"
        except Exception:
            pass
        # Try CSV
        try:
            df = pd.read_csv(io.StringIO(txt))
            return df, "Loaded CSV from pasted text"
        except Exception as e:
            return None, f"Error reading pasted text as CSV/JSON: {e}"

    return None, "Unsupported dataset input format."

def parse_agents_yaml(text_or_bytes):
    """
    Parse agents.yaml from file or pasted text.
    Returns: (agents_list, msg)
    Expected schema (flexible):
    agents:
      - name: ...
        description: ...
        visualization:
          type: bar|pie|scatter|hist
          x: column_name
          y: column_name
          hue: column_name
          aggregate: sum|count|mean (optional)
    """
    try:
        if isinstance(text_or_bytes, bytes):
            data = yaml.safe_load(io.StringIO(text_or_bytes.decode("utf-8")))
        else:
            data = yaml.safe_load(text_or_bytes)
    except Exception as e:
        return [], f"Error parsing YAML: {e}"

    if not data:
        return [], "Empty YAML content."

    agents = data.get("agents") or data.get("Agents") or data.get("AGENTS")
    if not agents or not isinstance(agents, list):
        return [], "No agents list found in YAML."

    cleaned = []
    for a in agents:
        name = a.get("name") or a.get("id") or a.get("title")
        viz = a.get("visualization") or {}
        cleaned.append({
            "name": name or "Unnamed Agent",
            "description": a.get("description", ""),
            "visualization": {
                "type": viz.get("type"),
                "x": viz.get("x"),
                "y": viz.get("y"),
                "hue": viz.get("hue"),
                "aggregate": viz.get("aggregate"),
            }
        })
    return cleaned, f"Parsed {len(cleaned)} agents."

# ============================================================
# 🤖 Gemini 2.0 Flash: Suggest visualization config
# ============================================================
def suggest_viz_with_gemini(api_key, df, agent_context=""):
    """
    Use Gemini 2.0 Flash to suggest a visualization config based on DF columns and agent context.
    Returns a dict like:
      {"type":"bar","x":"BrandName","y":"DeviceCount","hue":null,"aggregate":"sum"}
    """
    if not api_key:
        return None, "No API key provided for Gemini."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        cols = list(df.columns)
        sample = df.head(5).to_dict(orient="records")
        prompt = (
            "You are a data visualization assistant. Given the dataframe columns and a brief agent description, "
            "return a concise JSON with keys: type (bar|pie|scatter|hist), x, y, hue (optional), aggregate (sum|count|mean|none). "
            "Prioritize categorical x with numeric y for bar; use pie for categorical distribution; scatter for two numeric columns; hist for single numeric.\n\n"
            f"Columns: {cols}\nSample rows: {json.dumps(sample)}\nAgent context: {agent_context}\n"
            "Return only the JSON (no commentary)."
        )
        resp = model.generate_content(prompt)
        text = resp.text.strip()
        # Try to load JSON
        try:
            cfg = json.loads(text)
            return cfg, "Gemini suggested visualization."
        except Exception:
            # Robust fallback: try to extract between braces
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                cfg = json.loads(text[start:end+1])
                return cfg, "Gemini suggested visualization (parsed)."
            return None, "Could not parse Gemini response."
    except Exception as e:
        return None, f"Gemini error: {e}"

# ============================================================
# 📈 Visualization renderer
# ============================================================
def render_viz(df, viz_cfg):
    """
    Render a plot based on viz_cfg. Returns matplotlib figure.
    viz_cfg: {"type":"bar|pie|scatter|hist","x":..., "y":..., "hue":..., "aggregate":...}
    """
    fig = plt.figure(figsize=(8,5))
    vtype = (viz_cfg.get("type") or "").lower()
    x = viz_cfg.get("x")
    y = viz_cfg.get("y")
    hue = viz_cfg.get("hue")
    agg = (viz_cfg.get("aggregate") or "none").lower()

    # Simple aggregation support
    plot_df = df.copy()
    if agg in ["sum", "mean", "count"] and x and y and x in plot_df.columns and y in plot_df.columns:
        if agg == "count":
            plot_df = plot_df.groupby(x).size().reset_index(name="count")
            y = "count"
        else:
            plot_df = plot_df.groupby(x)[y].agg(agg).reset_index()

    # Fallback palette with coral
    palette = sns.color_palette([CORAL])

    try:
        if vtype == "bar" and x and y:
            sns.barplot(data=plot_df, x=x, y=y, hue=hue, color=CORAL if not hue else None, palette=None if not hue else "Set2")
            plt.title("Bar chart", fontsize=14)
            plt.xticks(rotation=45)
        elif vtype == "pie" and x:
            counts = plot_df[x].value_counts()
            plt.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=[CORAL, "#FFDAB9", "#FFC1A6", "#FFA07A"])
            plt.title("Pie chart", fontsize=14)
        elif vtype == "scatter" and x and y:
            sns.scatterplot(data=plot_df, x=x, y=y, hue=hue, palette="Set2")
            plt.title("Scatter plot", fontsize=14)
        elif vtype == "hist" and x:
            sns.histplot(data=plot_df, x=x, hue=hue, color=CORAL, bins=20)
            plt.title("Histogram", fontsize=14)
        else:
            plt.text(0.1, 0.5, f"Invalid or insufficient config for visualization.\nType: {vtype}\nX: {x}\nY: {y}", fontsize=12)
        plt.tight_layout()
        return fig
    except Exception as e:
        fig = plt.figure(figsize=(8,3))
        plt.text(0.1, 0.5, f"Visualization error: {e}", fontsize=12, color="red")
        plt.axis("off")
        return fig

# ============================================================
# 🖥️ Gradio UI: Upload/paste dataset & agents, select agent, visualize
# ============================================================
def run_pipeline(api_key, dataset_file, dataset_text, agents_file, agents_text, use_gemini):
    """
    Main function for Gradio.
    """
    # Load dataset
    ds_bytes = dataset_file.read() if dataset_file else None
    ds_name = dataset_file.name if dataset_file else None
    df, ds_msg = detect_format_and_load(ds_bytes if ds_bytes else dataset_text, filename=ds_name)

    # Load agents
    ag_bytes = agents_file.read() if agents_file else None
    agents, ag_msg = parse_agents_yaml(ag_bytes if ag_bytes else agents_text)

    status = []
    status.append(f"Dataset: {ds_msg}")
    status.append(f"Agents: {ag_msg}")

    agent_names = [a["name"] for a in agents] if agents else []
    return df, agents, "\n".join(status), gr.update(choices=agent_names, value=agent_names[0] if agent_names else None)

def visualize_selected(api_key, selected_agent_name, use_gemini, df_state, agents_state):
    """
    Visualize based on selected agent; optionally use Gemini to refine config.
    """
    if df_state is None or not isinstance(df_state, pd.DataFrame):
        msg = "No dataset loaded."
        fig = plt.figure(figsize=(8,3)); plt.text(0.1,0.5,msg); plt.axis("off")
        return fig, msg

    if not agents_state:
        msg = "No agents parsed."
        fig = plt.figure(figsize=(8,3)); plt.text(0.1,0.5,msg); plt.axis("off")
        return fig, msg

    # Find agent
    agent = None
    for a in agents_state:
        if a["name"] == selected_agent_name:
            agent = a
            break
    if agent is None:
        msg = f"Selected agent not found: {selected_agent_name}"
        fig = plt.figure(figsize=(8,3)); plt.text(0.1,0.5,msg); plt.axis("off")
        return fig, msg

    viz_cfg = agent.get("visualization") or {}
    # If config is minimal or missing, ask Gemini for suggestion
    gemini_msg = "Gemini not used."
    if use_gemini:
        suggested, gemini_msg = suggest_viz_with_gemini(api_key, df_state, agent_context=agent.get("description",""))
        if suggested:
            # Merge suggested into existing (agent overrides if present)
            for k,v in suggested.items():
                if viz_cfg.get(k) in [None, ""]:
                    viz_cfg[k] = v

    fig = render_viz(df_state, viz_cfg)
    # Compose status
    cfg_msg = json.dumps({k:v for k,v in viz_cfg.items()}, ensure_ascii=False)
    status = (
        f"Agent: {agent.get('name')}\n"
        f"Description: {agent.get('description','')}\n"
        f"Visualization config: {cfg_msg}\n"
        f"{gemini_msg}"
    )
    return fig, status

# ============================================================
# 🚀 Launch Gradio App
# ============================================================
with gr.Blocks(title="Agent-based Visualization (Colab + Gemini)") as demo:
    gr.Markdown("### Agent-based Data Visualization\nUpload or paste your dataset (JSON/CSV) and agents.yaml, then select an agent to visualize. Optional: use Gemini 2.0 Flash to auto-suggest visualization.")
    with gr.Row():
        api_key = gr.Textbox(label="Gemini API Key", placeholder="Enter your Gemini API key (optional if not using Gemini)", type="password")
        use_gemini = gr.Checkbox(label="Use Gemini 2.0 Flash for suggestions", value=True)

    with gr.Tab("Dataset"):
        dataset_file = gr.File(label="Upload dataset (CSV or JSON)")
        dataset_text = gr.Textbox(label="Or paste dataset (CSV or JSON)", lines=10, placeholder="Paste CSV or JSON here")

    with gr.Tab("Agents YAML"):
        agents_file = gr.File(label="Upload agents.yaml")
        agents_text = gr.Textbox(label="Or paste agents.yaml", lines=12, placeholder="Paste YAML with agents list")

    status = gr.Textbox(label="Status", interactive=False)
    df_state = gr.State()
    agents_state = gr.State()

    agent_dropdown = gr.Dropdown(choices=[], label="Select agent", interactive=True)

    load_btn = gr.Button("Load dataset & agents", variant="primary")
    viz_btn = gr.Button("Visualize", variant="secondary")

    load_btn.click(
        run_pipeline,
        inputs=[api_key, dataset_file, dataset_text, agents_file, agents_text, use_gemini],
        outputs=[df_state, agents_state, status, agent_dropdown],
    )

    plot = gr.Plot(label="Visualization")
    details = gr.Textbox(label="Details", interactive=False)

    viz_btn.click(
        visualize_selected,
        inputs=[api_key, agent_dropdown, use_gemini, df_state, agents_state],
        outputs=[plot, details],
    )

#demo.launch(share=True)
# The server_name="0.0.0.0" makes it accessible within the Render network.
# The server_port=7860 is a common port, but Render will map it to a public URL.
demo.launch(server_name="0.0.0.0", server_port=7860)
