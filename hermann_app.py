import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import os
import json
import ast
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.colors as mcolors
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode

# Inject external CSS
@st.cache_data
def load_css(file_name):
  with open(file_name) as f:
    return f"<style>{f.read()}</style>"


try:
  st.markdown(load_css("styles.css"), unsafe_allow_html=True)
except FileNotFoundError:
  pass  # Fail gracefully if file missing

API_URL = os.getenv("API_URL", st.secrets.get("API_URL"))
API_KEY = os.getenv("API_KEY", st.secrets.get("API_KEY"))

# hide Streamlit horrible menu
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    .stAppDeployButton {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)


icon = "assets/Favicon_2x.ico"
st.set_page_config(layout='wide', page_icon=icon, page_title="Outil d’analyse Boussole des personnalités")

st.image("assets/Banniere argios.svg")

st.title("Outil d'analyse pour le e-diagnostic des personnalités")

 # streamlit aggrid interactive dataframe
def aggrid_interactive_table(df: pd.DataFrame):
   """Creates an st-aggrid interactive table based on a dataframe.
   Args:
       df (pd.DataFrame]): Source dataframe
   Returns:
       dict: The selected row
   """
   options = GridOptionsBuilder.from_dataframe(
       df, enableRowGroup=True, enableValue=True, enablePivot=True
   )
   options.configure_side_bar()
   options.configure_selection(selection_mode='multiple',use_checkbox=True)
   selection = AgGrid(
       df,
       enable_enterprise_modules=True,
       fit_columns_on_grid_load=True,
       gridOptions=options.build(),
       theme="streamlit",
       update_mode=GridUpdateMode.SELECTION_CHANGED,
       allow_unsafe_jscode=True,
   )
   
   return selection

def fetch_results_from_database():
    url = f"{API_URL}/api/teams"
    headers = {
        "X-API-KEY": API_KEY,
        "Content-Type": "application/json",
    }
    response = requests.get(url, headers=headers)
    # print(response.status_code, response.text)
    response.raise_for_status()
    return response.json()

@st.cache_data(ttl=300)
def fetch_organizations_from_database():
    url = f"{API_URL}/api/organizations"
    headers = {
        "X-API-KEY": API_KEY,
        "Content-Type": "application/json",
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def post_orga_to_database(orga_name, description, status="ongoing"):
    url = f"{API_URL}/api/organizations"
    headers = {
        "X-API-KEY": API_KEY,
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }
    data = {
        "name": orga_name,
        "description": description,
        "status": status
    }
    response = requests.post(url, headers=headers, json=data)
    return response

def update_organization_status(orga_name, new_status):
    url = f"{API_URL}/api/organizations?name={orga_name}"
    headers = {
        "X-API-KEY": API_KEY,
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200 or not response.json():
        return response

    org_data = response.json()[0]
    org_id = org_data['id']
    org_data['status'] = new_status

    update_url = f"{API_URL}/api/organizations/{org_id}"
    headers["Prefer"] = "return=minimal"
    response = requests.put(update_url, headers=headers, json=org_data)
    return response

### Functions for compatibility graphs
# helper function to normalize user names (handles inconsistent naming conventions)
def normalize_name(person_label):
  """
  Normalize user names from various formats in the database.
  Handles: "Name-Surname", "Name-", "Name -", and other variations.
  Returns a clean name string suitable for display.
  """
  if pd.isna(person_label) or person_label == "":
    return "Unknown"

  person_str = str(person_label).strip()

  # Check if there's a hyphen (old naming convention)
  if "-" in person_str:
    parts = person_str.split("-")
    # Handle cases like "Name-", "Name -", etc.
    first = parts[0].strip() if parts else ""
    # Only include last name if it's not empty
    last = parts[1].strip() if len(parts) > 1 and parts[1].strip() else ""

    if last:
      return f"{first}-{last}" # Modified to use hyphen instead of newline
    else:
      return first
  else:
    # No hyphen found, return as is (new naming convention or simple name)
    return person_str
  
# helper function to extract just the name from PersonLabel
def extract_name(person_label):
  """Extract and format just the name from PersonLabel for display."""
  return normalize_name(person_label)

# helper function to extract and normalize name from dictionary keys
def extract_name_from_dict_key(name_with_artifacts):
  """
  Extract and normalize a name from dictionary key.
  Removes trailing/leading spaces and special characters (emojis, etc.)
  """
  import re
  # Strip whitespace
  cleaned = str(name_with_artifacts).strip()
  # Remove emojis and other special characters, keeping only alphanumeric, spaces, and common punctuation
  cleaned = re.sub(r'[^\w\s\-àâäéèêëïîôöùûüœæçÀÂÄÉÈÊËÏÎÔÖÙÛÜŒÆÇ]', '', cleaned)
  # Strip again in case emojis were at the edges
  cleaned = cleaned.strip()
  return cleaned

# helper function to force regular polygons
def polygon_layout(G, radius=3):
    """
    Return a dictionary of node positions arranged in a regular polygon.
    
    Args:
        G (nx.Graph): The graph with N nodes.
        radius (float): Radius of the polygon.
    
    Returns:
        dict: Node -> (x, y) position
    """
    nodes = list(G.nodes())
    n = len(nodes)
    angle_step = 2 * np.pi / n
    pos = {}

    for i, node in enumerate(nodes):
        angle = i * angle_step
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        pos[node] = (x, y)

    return pos

# function to create network graph with color-coded edges
def visualize_compatibility_network_colored(compatibility_matrix, threshold=1, figsize=(8, 8)):
    """
    Visualize compatibility using a networkx graph with color-coded edges and polygon layout.

    Args:
        compatibility_matrix (pd.DataFrame): Pairwise compatibility scores.
        threshold (float): Minimum score to draw an edge.
    """
    G = nx.Graph()

    # Add nodes
    for person in compatibility_matrix.index:
        G.add_node(person)

    # Add edges (only if score >= threshold)
    for i in compatibility_matrix.index:
        for j in compatibility_matrix.columns:
            if i != j:
                score = compatibility_matrix.loc[i, j]
                if score >= threshold:
                    G.add_edge(i, j, weight=score)

    # Use regular polygon layout
    pos = polygon_layout(G, radius=3.0)

    # Edge weights
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    edge_widths = [w / 1 for w in edge_weights]

    # Edge color map
    norm = mcolors.Normalize(vmin=threshold, vmax=5)
    colors = ['red', 'tomato', 'darkorange', 'orange', 'yellow','y', 'yellowgreen', 'limegreen', 'darkgreen']
    bounds = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

    # Create colormap and normalization
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    edge_colors = [cmap(norm(score)) for score in edge_weights]

    # Draw the network
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', edgecolors='black', node_size=1000)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors)
    # Use shortened names for labels if new convention, otherwise use the full label
    try:
        labels = {node: extract_name(node) for node in G.nodes()}
    except:
        labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_weight='bold', font_size=12)

    # Edge labels
    edge_labels = {(u, v): f"{G[u][v]['weight']:.0f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='black')

    # Colorbar legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, shrink=0.8, ax=ax)
    cbar.set_label('Compatibility Score')

    plt.title(f"Compatibility Network (Polygon Layout, Threshold ≥ {threshold})")
    plt.axis('off')
    plt.tight_layout()
    return fig

# New function: person-centered compatibility network
def visualize_person_centered_network_colored(compatibility_matrix, center_person, threshold=1, figsize=(8, 8)):
    """
    Visualize compatibility network with a selected person in the center.
    Only shows edges between center_person and others (no edges between other people).

    Args:
        compatibility_matrix (pd.DataFrame): Pairwise compatibility scores.
        center_person (str): The person to place in the center.
        threshold (float): Minimum score to draw an edge.
    """
    G = nx.Graph()
    # Add center node
    G.add_node(center_person)
    # Add other nodes and edges only to center_person
    for person in compatibility_matrix.index:
        if person != center_person:
            score = compatibility_matrix.loc[center_person, person]
            if score >= threshold:
                G.add_node(person)
                G.add_edge(center_person, person, weight=score)

    # Layout: center node at (0,0), others in a circle around
    n = len(G.nodes())
    pos = {}
    pos[center_person] = (0, 0)
    angle_step = 2 * np.pi / (n - 1) if n > 1 else 0
    radius = 3
    idx = 0
    for node in G.nodes():
        if node == center_person:
            continue
        angle = idx * angle_step
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        pos[node] = (x, y)
        idx += 1

    # Edge weights and colors
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    edge_widths = [w / 1 for w in edge_weights]
    colors = ['red', 'tomato', 'darkorange', 'orange', 'yellow','y', 'yellowgreen', 'limegreen', 'darkgreen']
    bounds = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    edge_colors = [cmap(norm(score)) for score in edge_weights]

    # Draw the network
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_color=['lightblue' if n != center_person else 'gold' for n in G.nodes()], edgecolors='black', node_size=1000)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors)
    # Use shortened names for labels if new convention, otherwise use the full label
    try:
        labels = {node: extract_name(node) for node in G.nodes()}
    except:
        labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_weight='bold', font_size=12)

    # Edge labels
    edge_labels = {(u, v): f"{G[u][v]['weight']:.0f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='black')

    # Colorbar legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, shrink=0.8, ax=ax)
    cbar.set_label('Compatibility Score')

    plt.title(f"Compatibilité centrée sur {center_person} (Seuil ≥ {threshold})")
    plt.axis('off')
    plt.tight_layout()
    return fig

# functions to create compatibility matrix
# helper function to get dominant quadrant
def get_dominant_quadrant(row, threshold=70):
    dominant = row.idxmax()
    if row[dominant] >= threshold:
        return dominant
    return None

def calculate_compatibility_scores(df, threshold=70, bridge_bonus=20, penalty=-30, base_score=50):
    # Ensure df only contains numeric A, B, C, D columns for scoring
    # (drop non-numeric columns that may have been included)
    score_cols = ['A', 'B', 'C', 'D']
    df_numeric = df[score_cols].copy()
    
    compatibility_matrix = pd.DataFrame(0.0, index=df_numeric.index, columns=df_numeric.index)

    for person1 in df_numeric.index:
        for person2 in df_numeric.index:
            if person1 == person2:
                compatibility_matrix.loc[person1, person2] = 100.0
                continue

            row1 = df_numeric.loc[person1]
            row2 = df_numeric.loc[person2]

            dom1 = get_dominant_quadrant(row1, threshold)
            dom2 = get_dominant_quadrant(row2, threshold)

            score = 0

            # Determine base compatibility
            incompatible_pairs = {
                ('A', 'C'): ('B', 'D'),
                ('C', 'A'): ('B', 'D'),
                ('D', 'B'): ('A', 'C'),
                ('B', 'D'): ('A', 'C')
            }

            if dom1 and dom2:
                if (dom1, dom2) in incompatible_pairs:
                    # Check for common bridge
                    bridges = incompatible_pairs[(dom1, dom2)]
                    shared_bridge = any(
                        row1[bridge] > threshold and row2[bridge] > threshold
                        for bridge in bridges
                    )
                    if shared_bridge:
                        score += base_score + bridge_bonus
                    else:
                        score += base_score + penalty
                else:
                    score += base_score

            # Add similarity component (cosine similarity of the 4-quadrant vectors)
            vec1 = np.array([row1[q] for q in ['A', 'B', 'C', 'D']]).reshape(1, -1)
            vec2 = np.array([row2[q] for q in ['A', 'B', 'C', 'D']]).reshape(1, -1)
            similarity = cosine_similarity(vec1, vec2)[0][0]  # range: 0 to 1
            score += similarity * 50  # scale similarity to 0–50

            # Clip score between 0 and 100
            compatibility_matrix.loc[person1, person2] = round(np.clip(score, 0, 100), 2)

    return compatibility_matrix


def calculate_compatibility_scores_on_five(df, threshold=70, bridge_bonus=20, penalty=-30, base_score=50):
    # Ensure df only contains numeric A, B, C, D columns for scoring
    # (drop non-numeric columns that may have been included)
    score_cols = ['A', 'B', 'C', 'D']
    df_numeric = df[score_cols].copy()
    
    compatibility_matrix = pd.DataFrame(0.0, index=df_numeric.index, columns=df_numeric.index)

    for person1 in df_numeric.index:
        for person2 in df_numeric.index:
            if person1 == person2:
                compatibility_matrix.loc[person1, person2] = 5.0
                continue

            row1 = df_numeric.loc[person1]
            row2 = df_numeric.loc[person2]

            dom1 = get_dominant_quadrant(row1, threshold)
            dom2 = get_dominant_quadrant(row2, threshold)

            score = 0

            # Determine base compatibility
            incompatible_pairs = {
                ('A', 'C'): ('B', 'D'),
                ('C', 'A'): ('B', 'D'),
                ('D', 'B'): ('A', 'C'),
                ('B', 'D'): ('A', 'C')
            }

            if dom1 and dom2:
                if (dom1, dom2) in incompatible_pairs:
                    # Check for common bridge
                    bridges = incompatible_pairs[(dom1, dom2)]
                    shared_bridge = any(
                        row1[bridge] > threshold and row2[bridge] > threshold
                        for bridge in bridges
                    )
                    if shared_bridge:
                        score += base_score + bridge_bonus
                    else:
                        score += base_score + penalty
                else:
                    score += base_score

            # Add similarity component (cosine similarity of the 4-quadrant vectors)
            vec1 = np.array([row1[q] for q in ['A', 'B', 'C', 'D']]).reshape(1, -1)
            vec2 = np.array([row2[q] for q in ['A', 'B', 'C', 'D']]).reshape(1, -1)
            similarity = cosine_similarity(vec1, vec2)[0][0]  # range: 0 to 1
            score += similarity * 50  # scale similarity to 0–50

            # Scale score from 0–100 to 1–5
            scaled_score = (np.clip(score, 0, 100) / 100) * 4 + 1
            compatibility_matrix.loc[person1, person2] = round(scaled_score, 2)

    return compatibility_matrix

# function to get team members with compatibility above a certain threshold
def dream_team(person, df, treshold):
  similarity_matrix = calculate_compatibility_scores_on_five(df)
  new = similarity_matrix.loc[similarity_matrix[person] > treshold]
  people = new.index.values.tolist()
  if person in people:
      people.remove(person)
      return people
  else:
      return people

# If the scores are stored in a single `evaluation` column (dict or JSON string with keys a/b/c/d),
# extract them into separate numeric columns `A`, `B`, `C`, `D` so existing code keeps working.
def _extract_scores_from_evaluation(eval_field):
    # Return a Series with A/B/C/D numeric values (or NaN when unavailable)
    if pd.isna(eval_field):
        return pd.Series({'A': np.nan, 'B': np.nan, 'C': np.nan, 'D': np.nan})

    parsed = None
    if isinstance(eval_field, dict):
        parsed = eval_field
    elif isinstance(eval_field, str):
        # Try JSON first, then fall back to Python literal eval (handles single quotes)
        try:
            parsed = json.loads(eval_field)
        except Exception:
            try:
                parsed = ast.literal_eval(eval_field)
            except Exception:
                parsed = None

    # Helper: try to find a dict that contains a/b/c/d (search nested structures)
    def _find_scores_dict(obj):
        if isinstance(obj, dict):
            keys_low = {k.lower() for k in obj.keys()}
            if any(k in keys_low for k in ('a', 'b', 'c', 'd')):
                return obj
            # common wrapper keys
            for wrapper in ('evaluation', 'scores', 'result', 'self'):
                if wrapper in obj and isinstance(obj[wrapper], dict):
                    return obj[wrapper]
            # search deeper
            for v in obj.values():
                found = _find_scores_dict(v)
                if found:
                    return found
        return None

    scores_dict = _find_scores_dict(parsed)
    if not isinstance(scores_dict, dict):
        # Could not locate a scores dict; return NaNs
        return pd.Series({'A': np.nan, 'B': np.nan, 'C': np.nan, 'D': np.nan})

    def _get_val(d, letter):
        # Prefer exact single-letter keys (case-insensitive), fall back to *_score style
        for key in (letter, letter.lower(), letter.upper()):
            if key in d:
                return d[key]
        for key in d.keys():
            kl = key.lower()
            if kl == f"{letter.lower()}_score" or (len(kl) == 1 and kl == letter.lower()):
                return d[key]
        return np.nan

    A = _get_val(scores_dict, 'A')
    B = _get_val(scores_dict, 'B')
    C = _get_val(scores_dict, 'C')
    D = _get_val(scores_dict, 'D')

    return pd.Series({
        'A': pd.to_numeric(A, errors='coerce'),
        'B': pd.to_numeric(B, errors='coerce'),
        'C': pd.to_numeric(C, errors='coerce'),
        'D': pd.to_numeric(D, errors='coerce'),
    })

# Radar chart helper
def radar_chart(name, scores, color='#6B5B95', team_avg=None):
    """
    Create a radar chart for HBDI profile.

    Args:
        name: Person's name for the title
        scores: List of 4 scores [A, B, C, D]
        color: Color for the individual profile line (default: purple #6B5B95)
        team_avg: Optional list of 4 scores [A, B, C, D] for team average overlay
    """
    # Create figure
    fig = go.Figure()

    # Add the individual's data trace with purple fill
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=['A', 'B', 'C', 'D'],
        fill='toself',
        fillcolor='rgba(107, 91, 149, 0.25)',  # Purple with 25% opacity
        name=name,
        line=dict(color=color, width=2.5),
        marker=dict(size=8, color=color)
    ))

    # Add team average as gray dashed line if provided
    if team_avg is not None:
        fig.add_trace(go.Scatterpolar(
            r=team_avg,
            theta=['A', 'B', 'C', 'D'],
            fill='none',
            name='Moyenne équipe',
            line=dict(color='gray', width=1.5, dash='dash'),
            opacity=0.7
        ))

    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showline=False,
                gridcolor='gray',
                tickfont=dict(size=14, weight='bold', color='gray'),
                gridwidth=0.5
            ),
            angularaxis=dict(
                tickfont=dict(size=14, weight='bold'),
                rotation=135,  # Rotated to put A in top left (not to upset hermann nerds)
                direction="clockwise",
                categoryorder='array',
                categoryarray=['A', 'D', 'C', 'B']  # Trigonometric
            )
        ),
        showlegend=team_avg is not None,  # Show legend only when team average is displayed
        title=f"Profil de {name}"
    )
    return fig

def calculate_team_average_similarity(team_members, similarity_matrix):
    """
    Calculates the average pairwise similarity score for a given team.

    Args:
        team_members: A list of strings representing the names of the people in the team.
        similarity_matrix: A pandas DataFrame containing the pairwise similarity scores.

    Returns:
        The average pairwise similarity score for the team, excluding self-similarity.
        Returns 0 if the team size is 1 or less.
    """
    n = len(team_members)
    if n <= 1:
        return 0.0

    total_similarity = 0
    num_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            person1 = team_members[i]
            person2 = team_members[j]
            total_similarity += similarity_matrix.loc[person1, person2]
            num_pairs += 1

    return total_similarity / num_pairs

# Create a unique label combining person name and ID to handle duplicate names
# Prefer including the unique `id` when available so the resulting index is unique.
def _make_person_label(row):
    name = row.get('Person', '')
    pid = row.get('id')
    if pd.notna(pid) and pid is not None:
        return f"{name} (ID:{pid})"
    return f"{name}"

##########################################
############# Main app logic #############
##########################################

data = fetch_results_from_database()

organisations = [row["organisation"] for row in data]
unique_organisations = list(set(organisations))
selected_organisation = st.selectbox("Select organisation", unique_organisations)

raw_df = pd.DataFrame([row for row in data if row["organisation"] == selected_organisation])
# Basic rename (scores are now stored in a single column like a dict/JSON under `evaluation`)
raw_df.rename(columns={"user": "Person", "organisation": "Organisation"}, inplace=True)

# Option to use all data or filter with aggrid
use_all_data = st.checkbox("Utiliser tous les données (sans sélection)", value=True)

if use_all_data:
    df = raw_df
    st.write("Équipe sélectionnée (tous les membres):")
else:
    filtered_df = aggrid_interactive_table(df=raw_df)
    df = pd.DataFrame(filtered_df['selected_rows'])
    st.write("Équipe sélectionnée:")
st.dataframe(data=df)
st.download_button(
            label="Télécharger le dataframe",
            data=df.to_csv(index=True).encode('utf-8'),
            file_name=f'Equipe_{selected_organisation}.csv',
            mime='text/csv',
        )

if df is None or len(df) == 0:
    st.warning("Veuillez séléctionner les personnes concernés par l'analyse dans le tableau ci-dessus.")

else:
    # Apply extraction if a scores column exists (try 'personalScore', 'evaluation', etc.). If A/B/C/D already exist keep them (prefer non-null values).
    scores_column = None
    for col_name in ['personalScore', 'evaluation', 'scores']:
        if col_name in df.columns:
            scores_column = col_name
            break

    if scores_column:
        scores_df = df[scores_column].apply(_extract_scores_from_evaluation)
    else:
        scores_df = pd.DataFrame(index=df.index, columns=['A','B','C','D'])

    for col in ['A','B','C','D']:
        if col in df.columns:
            # keep existing numeric values and fill missing from parsed scores
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(scores_df[col])
        else:
            df[col] = scores_df[col]

    # Do not auto-fill NaNs here — keep NaNs so parsing issues are visible and can be handled upstream.
    # If you prefer to replace NaNs with 0 for plotting, uncomment the next line:
    #df[['A','B','C','D']] = df[['A','B','C','D']].fillna(0)
    # Normalize person names to handle inconsistent naming conventions from database
    df['Person'] = df['Person'].apply(lambda x: normalize_name(x) if pd.notna(x) else x)

    df["PersonLabel"] = df.apply(_make_person_label, axis=1)
    # Set the PersonLabel as index (which is now unique)
    df = df.set_index("PersonLabel")
    # Keep the original person name and ID as columns for reference
    df['PersonName'] = df['Person']
    df['PersonID'] = df['id']
    # Short canonical name (used for matching with external keys / evaluation dicts)
    df['PersonShort'] = df['PersonName'].apply(lambda x: extract_name_from_dict_key(x))
    #st.dataframe(data=df)
    #df["Dominant"] = df.idxmax(axis=1)
    #st.subheader("🧪 Raw Data Preview")
    #st.dataframe(data=df)
    # Sidebar for user interaction

    selected_view = st.sidebar.radio("Choisir une analyse:", ["Profil individuel", "Profil moyen de l'équipe", "Styles dominants", "Déviations", "Les autres", "Compatibilité", "Communication", "Team maker", "Database operations"])


    # View 1: Individual radar chart
    if selected_view == "Profil individuel":
        # Calculate team average for comparison overlay
        team_avg_scores = df[["A", "B", "C", "D"]].mean().tolist()

        selected_person = st.selectbox("Sélectionner un membre de l'équipe:", df.index.tolist())
        if selected_person:
            person_scores = df.loc[selected_person][["A", "B", "C", "D"]].tolist()
            fig = radar_chart(selected_person, person_scores, team_avg=team_avg_scores)
            st.plotly_chart(fig)
        else:
            st.warning("Veuillez sélectionner un membre de l'équipe.")

        # Button to save all radar charts to team_graphs folder
        if st.button("Send to markdown"):
            output_folder = "team_graphs"
            os.makedirs(output_folder, exist_ok=True)

            saved_files = []
            for person in df.index.tolist():
                person_scores = df.loc[person][["A", "B", "C", "D"]].tolist()
                fig = radar_chart(person, person_scores, team_avg=team_avg_scores)
                # Sanitize filename by replacing problematic characters
                safe_name = str(person).replace("/", "_").replace("\\", "_").replace(":", "_").replace("(", "-").replace(")", "-")
                filepath = os.path.join(output_folder, f"{safe_name}.png")
                fig.write_image(filepath)
                saved_files.append((person, f"{safe_name}.png"))

            # Update the markdown file with the generated plots
            markdown_file = "cle_lecture1.md"
            try:
                with open(markdown_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Find the section to replace
                section_start = "### Résultats de notre équipe :"
                section_end = "## Synthèse des pilotes de personnalité"

                if section_start in content and section_end in content:
                    start_idx = content.index(section_start)
                    end_idx = content.index(section_end)

                    # Build new image references (use name without ID for alt text)
                    image_refs = "\n".join([f"![{person.split('(')[0].strip()}]({output_folder}/{filename})" for person, filename in saved_files])
                    new_section = f"{section_start}\n\n{image_refs}\n\n"

                    # Replace the section
                    new_content = content[:start_idx] + new_section + content[end_idx:]

                    with open(markdown_file, 'w', encoding='utf-8') as f:
                        f.write(new_content)

                    st.success(f"Graphiques sauvegardés dans '{output_folder}' et '{markdown_file}' mis à jour!")
                    st.download_button(
                        label="Télécharger le fichier markdown",
                        data=new_content.encode('utf-8'),
                        file_name=markdown_file,
                        mime='text/markdown',)
                else:
                    st.warning(f"Section non trouvée dans {markdown_file}. Graphiques sauvegardés uniquement.")
            except FileNotFoundError:
                st.warning(f"Fichier {markdown_file} non trouvé. Graphiques sauvegardés uniquement.")


    # View 2: Profil moyen de l'équipe
    elif selected_view == "Profil moyen de l'équipe":
        avg_scores = df[["A", "B", "C", "D"]].mean().tolist()
        fig = radar_chart("Profil moyen de l'équipe", avg_scores)
        st.plotly_chart(fig)

    # View 4: Styles dominants
    elif selected_view == "Styles dominants":
        # Calculate dominant style for each person
        df['Dominant'] = df[['A', 'B', 'C', 'D']].idxmax(axis=1)
        
        st.subheader("Mode de pensée dominant par personne")
        dominant_df = df[['Dominant']].copy()
        
        # Debug: Show raw dominant values for each person
        #st.write("Raw dominant values per person:")
        #st.write(df[['A', 'B', 'C', 'D']])
        st.write("Quadrant dominant calculé par personne:")
        st.dataframe(data=df['Dominant'])
        st.download_button(
            label="Télécharger les modes de pensée dominants",
            data=df['Dominant'].to_csv(index=True).encode('utf-8'),
            file_name='hermann_dominant_styles.csv',
            mime='text/csv',
        )
        
        # Create pie chart of dominant styles distribution
        dominant_counts_series = df['Dominant'].value_counts()
        
        # For debugging/displaying the counts
        dominant_counts_df_display = dominant_counts_series.reset_index()
        dominant_counts_df_display.columns = ['Quadrant', 'Nombre de personnes']

        # Get values and names as lists
        df_values = dominant_counts_df_display['Nombre de personnes'].tolist()
        df_names = dominant_counts_df_display['Quadrant'].tolist()

        # Define colors for each quadrant
        quadrant_colors = {
            'A': 'royalblue',
            'B': 'green',
            'C': 'firebrick',
            'D': 'gold'
        }

        # Create a list of colors corresponding to the order of df_names
        pie_colors = [quadrant_colors[name] for name in df_names]

        # Create pie chart using matplotlib
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.pie(
            df_values,
            labels=df_names,
            colors=pie_colors,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'edgecolor':'white', 'width':0.7},
        )
        ax.axis('equal')
        ax.set_title("Distribution of Dominant Quadrant", color='white', fontsize=12)
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(fig)
        with col2:
            st.dataframe(dominant_counts_df_display)

    # View 5: Déviations
    elif selected_view == "Déviations":
        # Heatmap - only keep numeric score columns for normalization
        df_scores = df[["A", "B", "C", "D"]].copy()

        #st.dataframe(data=df_scores)

        # Standardize (Z-score) so deviations stand out
        df_norm = (df_scores - df_scores.mean()) / df_scores.std()

        # Reset index to preserve PersonLabel as a column, then melt
        df_melt = df[["A", "B", "C", "D"]].reset_index()
        melted = df_melt.melt(id_vars="PersonLabel", value_vars=["A", "B", "C", "D"], var_name="Quadrant", value_name="Score")
        # this same data can be used in the boxplot
        quadrant_data = [
            melted[melted['Quadrant'] == 'A']['Score'].values,
            melted[melted['Quadrant'] == 'B']['Score'].values,
            melted[melted['Quadrant'] == 'C']['Score'].values,
            melted[melted['Quadrant'] == 'D']['Score'].values
        ]

        quadrant_labels = ['A', 'B', 'C', 'D']

        figsize_diverge1 = st.slider("Ajuster la hauter pour la heatmap des déviations (Z-score)", min_value=6, max_value=25, value=8, step=1)
        figsize_diverge2 = st.slider("Ajuster la largeur pour la heatmap des déviations (Z-score)", min_value=6, max_value=25, value=8, step=1)
        fig_diverge = plt.figure(figsize=(figsize_diverge1, figsize_diverge2))
        fig_diverge, ax_diverge = plt.subplots(figsize=(figsize_diverge1, figsize_diverge2))
        sns.heatmap(df_norm, annot=df_scores, cmap="coolwarm", center=0, cbar_kws={'label': 'Z-score'}, fmt='.1f', ax=ax_diverge)
        # Set shortened names for y-axis labels
        # extract names if old convention otherwise dont' bother
        try:
            ax_diverge.set_yticklabels([extract_name(label.get_text()) for label in ax_diverge.get_yticklabels()])
        except:
            ax_diverge.set_yticklabels([label.get_text() for label in ax_diverge.get_yticklabels()])
        st.write("Valeurs divergeantes mis en évidence par couleur")
        col_div1, col_div2 = st.columns(2)
        with col_div1:
            st.pyplot(fig_diverge)
        with col_div2:
            st.markdown("""Légende \n:
    - **Chaque case** représente le score d’un membre de l’équipe pour un quadrant (A, B, C, D), standardisé (Z-score) par rapport à la moyenne de l’équipe.
    - **Couleurs chaudes (rouge/rose)** : Score supérieur à la moyenne de l’équipe pour ce quadrant.
    - **Couleurs froides (bleu)** : Score inférieur à la moyenne de l’équipe pour ce quadrant.
    - **Blanc ou proche du gris** : Score proche de la moyenne de l’équipe.
    - **Valeur affichée** : Score brut du membre pour ce quadrant.""")
            #st.subheader("Scores moyennes par quandrant")
            meanz = pd.DataFrame(df[["A", "B", "C", "D"]].mean(), columns=["Valeur moyenne dans l'équipe"])
            #st.dataframe(data=meanz)
            # Calculate outlier boundaries and identify outliers
            outlier_info = []
            for quadrant in quadrant_labels:
                scores = melted[melted['Quadrant'] == quadrant]['Score']
                Q1 = scores.quantile(0.25)
                Q3 = scores.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

            # Find persons with scores outside the bounds
            for index, row in df.iterrows():
                score = row[quadrant]
                if score < lower_bound or score > upper_bound:
                    outlier_info.append(f"- '{index}' a un score divergeant {score} pour le Quadrant {quadrant}")

            if outlier_info:
                for info in outlier_info:
                    st.write(info)
            else:
                st.write(":cherry_blossom: ***Aucune valeur divergeante détectée***")


        # Dumbbell Plot or Slope Chart (Person vs. Team Average)
        st.subheader("Deviation par rapport à la moyenne de l'équipe par Quadrant")
        st.write("Score de la personne et la déviation par rapport à la moyenne de l'équipe")
        categories = ["A", "B", "C", "D"]
        avg_scores = df_scores.mean()
        div_figsize1 = st.slider("Ajuster la hauter pour le graphique des déviations", min_value=4, max_value=20, value=6, step=1)
        div_figsize2 = st.slider("Ajuster la largeur pour le graphique des déviations", min_value=8, max_value=25, value=12, step=1)
        fig, axs = plt.subplots(1, 4, figsize=(div_figsize2, div_figsize1), sharey=True)
        # Create shortened labels for y-axis
        shortened_labels = [label for label in df_scores.index]
        for i, cat in enumerate(categories):
            axs[i].hlines(y=df_scores.index, xmin=avg_scores[cat], xmax=df_scores[cat], color='grey', alpha=0.5)
            axs[i].plot(df_scores[cat], df_scores.index, "o", label="Score")
            axs[i].vlines(avg_scores[cat], 0, len(df_scores), color="red", linestyles="dashed", label="Moyenne")
            axs[i].set_title(f"Quadrant {cat}")
            axs[i].set_xlabel("Score")
            # Set shortened y-axis labels
            axs[i].set_yticks(range(len(df_scores.index)))
            axs[i].set_yticklabels(shortened_labels)
            if i == 0:
                axs[i].legend()
        st.pyplot(fig)
        
        # Create box plot using matplotlib
    #    fig, ax = plt.subplots(figsize=(8, 6))
    #    ax.boxplot(quadrant_data, labels=quadrant_labels)
    #
    #    ax.set_title("Distribution des scores dans l'équipe")
    #    ax.set_xlabel("Quadrant")
    #    ax.set_ylabel("Score")
    #    ax.set_ylim(0, 125) # Set y-axis limits
    #    ax.grid(axis='y', linestyle='--', alpha=0.7) # Add a grid

    #   st.pyplot(fig)

    elif selected_view == "Compatibilité":
        st.subheader("Analyse de compatibilité entre les membres de l'équipe")
        df_scores = df.drop(columns=["PersonLabel", "Organisation", "evaluation", "Person", "id", "PersonName", "PersonID"], errors='ignore')
        # With unique index, drop_duplicates should not be needed, but kept for safety
        st.download_button(
            label="Télécharger les scores bruts",
            data=df_scores.to_csv(index=True).encode('utf-8'),
            file_name='hermann_scores.csv',
            mime='text/csv',
        )
        # Calculate compatibility scores
        similarity_matrix = calculate_compatibility_scores_on_five(df_scores)
        #st.dataframe(data=similarity_matrix)
        minim_comp = similarity_matrix.min().min()
        index_min = similarity_matrix.idxmin()
        st.subheader("Réseau de compatibilité")
        treshold = st.number_input("Enrez le seuil de compatibilité pour afficher les connexions (1-5):", min_value=1.0, max_value=5.0, value=3.0, step=0.1, key="compatibility_threshold")
        figsize1 = st.slider("Hauter de la figure:", min_value=5, max_value=15, value=8, step=1, key="compatibility_figsize1")
        figsize2 = st.slider("Largeur de la figure:", min_value=5, max_value=15, value=8, step=1, key="compatibility_figsize2")
        figsize = (figsize1, figsize2)
        fig = visualize_compatibility_network_colored(similarity_matrix, threshold=treshold, figsize=figsize)
        st.pyplot(fig)
        st.markdown("""
    * Chaque nœud représente un membre de l'équipe.
    * Les arêtes représentent la compatibilité entre les membres.
    * La largeur des arêtes est proportionnelle au score de compatibilité.
    * Les couleurs des arêtes vont du rouge (faible compatibilité) au vert (forte compatibilité).
    * Le score de compatibilité est indiqué par le texte sur les arêtes""")
        st.subheader("Matrice de compatibilité")
        matrix_size1 = st.slider("Hauter de la figure:", min_value=5, max_value=15, value=6, step=1, key="compatibility_figsize3")
        matrix_size2 = st.slider("Largeur de la figure:", min_value=5, max_value=15, value=6, step=1, key="compatibility_figsize4")
        matrix_figsize = (matrix_size1, matrix_size2)
        plt.figure(figsize=matrix_figsize)
        fig_comp, ax_comp = plt.subplots(figsize=matrix_figsize)
        sns.heatmap(similarity_matrix, annot=True, cmap='BuPu', fmt=".1f", ax=ax_comp)
        # Set shortened names for both axes if new convention
        try:
            ax_comp.set_xticklabels([extract_name(label.get_text()) for label in ax_comp.get_xticklabels()], rotation=45, ha='right')
        except:
            ax_comp.set_xticklabels([label.get_text() for label in ax_comp.get_xticklabels()], rotation=45, ha='right')
        try:
            ax_comp.set_yticklabels([extract_name(label.get_text()) for label in ax_comp.get_yticklabels()])
        except:
            ax_comp.set_yticklabels([label.get_text() for label in ax_comp.get_yticklabels()])
        st.pyplot(fig_comp)
        st.markdown("""
    * Chaque case représente le score de compatibilité entre deux membres de l'équipe.
    * Les scores vont de 0 (aucune compatibilité) à 5 (compatibilité parfaite).""")
            
        # Dream team
        st.subheader("Créer des sous équipes selon la compatibilité")
        df['Dominant'] = df[['A', 'B', 'C', 'D']].idxmax(axis=1)
        dominant_df = df[['Dominant']].copy()
        #st.dataframe(dominant_df)
        target_person = st.selectbox("Sélectionner un membre de l'équipe pour créer une sous-équipe:", df_scores.index.tolist(), key="dream_team_person")
        threshold = st.slider("Seuil de compatibilité (1-5):", min_value=1.0, max_value=5.0, value=3.0, step=0.1, key="dream_team_threshold")
        if target_person:
            team_members = dream_team(target_person, df_scores, threshold)
            if team_members:
                st.write(f"Membres de l'équipe compatibles avec {target_person} (seuil ≥ {threshold}):")
                for member in team_members:
                    st.write(f"- {member} (Quadrant dominant: {df.loc[member, 'Dominant']})")
            else:
                st.write(f"Aucun membre de l'équipe n'a une compatibilité ≥ {threshold} avec {target_person}.")

        # Dream team with a given number of members
        st.subheader("Créer une sous-équipe avec un nombre défini de membres")
        average_similarity_scores = similarity_matrix.mean(axis=1).sort_values(ascending=False)
        #st.write(average_similarity_scores)

        dream_team_size = st.slider("Taille de la sous-équipe:", min_value=2, max_value=df.shape[0], value=4, step=1, key="dream_team_size")
        n = dream_team_size
        # Initial team: top n based on average similarity
        initial_team = average_similarity_scores.head(n).index.tolist()
        candidate_teams = [initial_team]

        # Generate more candidate teams: teams centered around top individuals and their most compatible peers
        top_people = average_similarity_scores.head(5).index.tolist() # Consider top 5 for centering
        for person in top_people:
            # Find n-1 most compatible people for this person, excluding self
            most_compatible = similarity_matrix.loc[person].drop(person).sort_values(ascending=False).head(n-1).index.tolist()
            candidate_team = [person] + most_compatible
            candidate_teams.append(candidate_team)

        # Remove duplicate teams (if any)
        unique_candidate_teams = []
        for team in candidate_teams:
            if sorted(team) not in [sorted(t) for t in unique_candidate_teams]:
                unique_candidate_teams.append(team)

        print("Generated Candidate Teams:")
        for i, team in enumerate(unique_candidate_teams):
            st.write(f"Team {i+1}: {team}")

        best_team = None
        best_average_similarity = -1

        for team in unique_candidate_teams:
            current_average_similarity = calculate_team_average_similarity(team, similarity_matrix)
            if current_average_similarity > best_average_similarity:
                best_average_similarity = current_average_similarity
                best_team = team

        st.subheader(f"Meilleure sous-équipe de taille {n} basée sur la compatibilité moyenne")

        st.write(f"l'équipe la plus compatible: {best_team}")
        st.write(f"Score max de compatibilité (de cette équipe): {best_average_similarity}")
        new_matrix = similarity_matrix.filter(best_team)
        new_matrix.reset_index(inplace=True)
        # Get the column name from the index (will be 'PersonLabel' or the index name)
        index_col_name = new_matrix.columns[0]  # First column after reset_index is the former index
        team_matrix = new_matrix[new_matrix[index_col_name].isin(best_team)]
        #st.dataframe(data=team_matrix)
        team_matrix.set_index(index_col_name, inplace=True)
        fig_team = visualize_compatibility_network_colored(team_matrix)
        st.pyplot(fig_team)


    elif selected_view == "Les autres":
        # Calculate team average for comparison overlay in self-evaluation charts
        team_avg_scores = df[["A", "B", "C", "D"]].mean().tolist()

        # Show shortened names but keep the full PersonLabel as the returned value so we can index df safely
        people_to_select = df.index.tolist()
        selected_person = st.selectbox("Sélectionner un membre de l'équipe:", people_to_select) #I could add a format_func here to shorten names if need be...
        if selected_person:
            st.header(f"L'équipe selon {selected_person}")

            # selected_person is the full PersonLabel (e.g. 'Name (ID: 123)'), so indexing df with it is safe
            person_evaluations_data = df.loc[selected_person, "evaluation"]

            if person_evaluations_data is None or pd.isna(person_evaluations_data):
                st.info(f"Pas de données d'évaluation croisée pour {selected_person}.")
            else:
                person_evaluations_dict = None
                if isinstance(person_evaluations_data, str):
                    try:
                        person_evaluations_dict = json.loads(person_evaluations_data)
                    except json.JSONDecodeError:
                        st.error("Impossible de parser les données d'évaluation (JSON mal formaté).")
                        st.write("Données reçues:", person_evaluations_data)
                elif isinstance(person_evaluations_data, dict):
                    person_evaluations_dict = person_evaluations_data

                if not person_evaluations_dict or not isinstance(person_evaluations_dict, dict):
                    st.warning(f"Les données d'évaluation pour {selected_person} ne sont pas dans un format correct (attendu: dictionnaire).")
                    st.write("Données reçues:", person_evaluations_data)
                else:
                    #st.write(person_evaluations_dict.keys())
                    names = set()
                    for key in person_evaluations_dict.keys():
                        parts = key.split('_', 1)
                        if len(parts) == 2:
                            quadrant, name = parts
                            if quadrant in ['A', 'B', 'C', 'D']:
                                # Normalize the name: strip spaces, remove emojis, etc.
                                normalized_name = extract_name_from_dict_key(name)
                                names.add((normalized_name, name))  # Store both normalized and original

                    if not names:
                        st.warning(f"Le dictionnaire d'évaluation pour {selected_person} est vide ou ne contient pas de données exploitables.")
                        st.write(person_evaluations_dict)
                    else:
                        for normalized_name, original_name in sorted(list(names), key=lambda x: x[0]):
                            col1, col2 = st.columns(2)

                            with col1:
                                scores_by_other = [
                                    person_evaluations_dict.get(f'A_{original_name}', 0) * 25,
                                    person_evaluations_dict.get(f'B_{original_name}', 0) * 25,
                                    person_evaluations_dict.get(f'C_{original_name}', 0) * 25,
                                    person_evaluations_dict.get(f'D_{original_name}', 0) * 25
                                ]
                                fig_by_other = radar_chart(normalized_name, scores_by_other)
                                # Display short name and indicate who provided the evaluation
                                title_by_other = f"Profil de {normalized_name}<br>(vu par {selected_person})"
                                fig_by_other.update_layout(title=dict(text=title_by_other, font=dict(size=16)))
                                st.plotly_chart(fig_by_other)

                            with col2:
                                # Match by normalized name in the dataframe index
                                # The df index contains normalized names (after the normalize_name() call at data load)
                                # Match by the short canonical name (ignores the appended ID in the index)
                                matched_labels = df[df['PersonShort'] == normalized_name].index.tolist()
                                if not matched_labels:
                                    st.warning(f"Le profil de {normalized_name} n'a pas pu être trouvé.")
                                elif len(matched_labels) == 1:
                                    matched_label = matched_labels[0]
                                    real_scores = df.loc[matched_label, ["A", "B", "C", "D"]].tolist()
                                    # Show the short name but use the full label for context (ID)
                                    title_real = f"Profil de {normalized_name} (auto-évaluation — {matched_label})"
                                    fig_real = radar_chart(normalized_name, real_scores, team_avg=team_avg_scores)
                                    fig_real.update_layout(title=dict(text=title_real, font=dict(size=16)))
                                    st.plotly_chart(fig_real)
                                else:
                                    # Multiple people share the same short name: show all matches
                                    for matched_label in matched_labels:
                                        real_scores = df.loc[matched_label, ["A", "B", "C", "D"]].tolist()
                                        title_real = f"Profil de {normalized_name} (auto-évaluation — {matched_label})"
                                        fig_real = radar_chart(normalized_name, real_scores, team_avg=team_avg_scores)
                                        fig_real.update_layout(title=dict(text=title_real, font=dict(size=14)))
                                        st.plotly_chart(fig_real)
        else:
            st.warning("Veuillez sélectionner un membre de l'équipe.")

    elif selected_view == "Communication":
        st.subheader("Les modes de communication à privilégier dans l'équipe")
        comm_people = df.index.tolist()
        comm_person = st.selectbox("Sélectionner un membre de l'équipe:", comm_people, key="comm_person")
        df_scores = df.drop(columns=["PersonLabel", "Organisation", "evaluation", "Person", "id", "PersonName", "PersonID"], errors='ignore')
        # With unique index, drop_duplicates should not be needed, but kept for safety
        # Calculate compatibility scores
        similarity_matrix = calculate_compatibility_scores_on_five(df_scores)
        #st.dataframe(data=similarity_matrix)
        minim_comp = similarity_matrix.min().min()
        index_min = similarity_matrix.idxmin()
        #st.write(minim_comp)
        st.subheader("Réseau de compatibilité")
        treshold = st.number_input("Enrez le seuil de compatibilité pour afficher les connexions (1-5):", min_value=1.0, max_value=5.0, value=3.0, step=0.1, key="comm_compatibility_threshold")
        figsize1 = st.slider("Hauter de la figure:", min_value=5, max_value=15, value=8, step=1, key="comm_compatibility_figsize1")
        figsize2 = st.slider("Largeur de la figure:", min_value=5, max_value=15, value=8, step=1, key="comm_compatibility_figsize2")
        figsize = (figsize1, figsize2)
        fig = visualize_person_centered_network_colored(similarity_matrix, comm_person, threshold=treshold, figsize=figsize)
        st.pyplot(fig)
        st.markdown("""
    * Chaque nœud représente un membre de l'équipe.
    * Les arêtes représentent la compatibilité entre les membres.
    * La largeur des arêtes est proportionnelle au score de compatibilité.
    * Les couleurs des arêtes vont du rouge (faible compatibilité) au vert (forte compatibilité).
    * Le score de compatibilité est indiqué par le texte sur les arêtes""")
        st.subheader("Modes de communication à privilégier")
        target_people = comm_people.remove(comm_person)
        df['Dominant'] = df[['A', 'B', 'C', 'D']].idxmax(axis=1)  
        dominant_df = df[['Dominant']].copy()
        st.markdown(f"**{comm_person}** est pilote : {df.loc[comm_person, 'Dominant']}")
        for person in comm_people:
            if df.loc[person, 'Dominant'] == "A":
                comm_style = "Numérico-analytique, structuré, logique, factuel"
            elif df.loc[person, 'Dominant'] == "B":
                comm_style = "Organisé, planifié, méthodique, précis"
            elif df.loc[person, 'Dominant'] == "C":
                comm_style = "Communicatif, relationnel, conversations"
            elif df.loc[person, 'Dominant'] == "D":
                comm_style = "Schématique, sythètique, vision globale, idées"
            score = similarity_matrix.loc[comm_person, person]
            if score >= 4.5:
                comm_quality = "Communication fluide et naturelle"
            elif score >= 3.5:
                comm_quality = "Communication efficace avec quelques ajustements"
            elif score >= 2.5:
                comm_quality = "Communication possible mais nécessite des efforts conscients"
            elif score >= 1.5:
                comm_quality = "Communication difficile, nécessite une adaptation significative"
            else:
                comm_quality = "Communication très difficile, risque de malentendus fréquents"
            st.markdown(f"""
    - **{person}** (score: {score}): *{comm_quality}*. Avec {person} il faut privilégier : {comm_style}.""")

    elif selected_view == "Team maker":
        st.subheader("Feuille de match")

        uploaded_files = st.file_uploader(
            "Upload csv", accept_multiple_files=True, type="csv"
        )

        if not uploaded_files:
            st.info("Veuillez uploader un fichier CSV contenant les colonnes: Person, Role, A, B, C, D")
        else:
            tm_frames = []
            for uploaded_file in uploaded_files:
                tm_frame = pd.read_csv(uploaded_file)
                tm_frames.append(tm_frame)
                st.dataframe(data=tm_frame)
            tm_df = pd.concat(tm_frames, ignore_index=True)

            if "Role" not in tm_df.columns or "Person" not in tm_df.columns:
                st.error("Le CSV doit contenir les colonnes 'Person' et 'Role'.")
            else:
                st.header("Rôles")
                roles = tm_df["Role"].unique().tolist()
                selected_roles = st.multiselect("Select roles to include:", roles, default=roles)

                selected_people = []
                for role in selected_roles:
                    people_in_role = tm_df[tm_df["Role"] == role]["Person"].tolist()
                    selected = st.multiselect(f"Select people for role: {role}", people_in_role)
                    selected_people.extend(selected)

                st.write(f"Selected people: {selected_people}")

                if "tm_show_network" not in st.session_state:
                    st.session_state.tm_show_network = False
                if "tm_compatibility_matrix" not in st.session_state:
                    st.session_state.tm_compatibility_matrix = None
                if "tm_network_selected_people" not in st.session_state:
                    st.session_state.tm_network_selected_people = []

                if st.button("Create network graph"):
                    if not selected_people:
                        st.warning("Veuillez sélectionner au moins une personne.")
                    else:
                        df_selected = tm_df[tm_df["Person"].isin(selected_people)].set_index("Person")
                        st.session_state.tm_compatibility_matrix = calculate_compatibility_scores_on_five(df_selected)
                        st.session_state.tm_network_selected_people = selected_people.copy()
                        st.session_state.tm_show_network = True

                if st.session_state.tm_show_network and st.session_state.tm_compatibility_matrix is not None:
                    compatibility_matrix = st.session_state.tm_compatibility_matrix
                    network_people = st.session_state.tm_network_selected_people

                    st.subheader("Compatibility Matrix")
                    st.dataframe(compatibility_matrix)

                    st.subheader("Compatibility Network Graph")
                    tm_threshold = st.slider("Select compatibility threshold for edges:", min_value=1, max_value=5, value=3, step=1, key="tm_threshold")
                    tm_figsize1 = st.slider("Hauteur de la figure:", min_value=5, max_value=15, value=8, step=1, key="tm_figsize1")
                    tm_figsize2 = st.slider("Largeur de la figure:", min_value=5, max_value=15, value=8, step=1, key="tm_figsize2")
                    fig = visualize_compatibility_network_colored(compatibility_matrix, threshold=tm_threshold, figsize=(tm_figsize1, tm_figsize2))
                    st.pyplot(fig)

                    st.subheader("Person-Centered Compatibility Networks")
                    center_person = st.selectbox("Select a person to center the network on:", network_people, key="tm_center_person")
                    tm_threshold2 = st.slider("Select compatibility threshold for centered network edges:", min_value=1, max_value=5, value=3, step=1, key="tm_centered_threshold")
                    fig_centered = visualize_person_centered_network_colored(compatibility_matrix, center_person, threshold=tm_threshold2, figsize=(tm_figsize1, tm_figsize2))
                    st.pyplot(fig_centered)

    elif selected_view == "Database operations":
        st.subheader("Organization Operations")

        orga_data = fetch_organizations_from_database()
        st.write("### Organizations table from database:")
        st.dataframe(data=orga_data)

        list_of_orga = [item["name"] for item in orga_data]

        st.subheader("Create new organization:")
        orga_name = st.text_input("e-diag id (ex: Adeo ou handball_team)")
        st.write(f"e-diag id will appear as: {orga_name}")

        description = st.text_input("description of the e-diag campaign (ex: what is the purpose of this e-diag? Who is responsible?)")
        st.write(f"description de l'e-diag: {description}")

        if orga_name in list_of_orga:
            st.error("cet id existe déjà")

        submit_button = st.button("Générer l'e-diag")
        if submit_button:
            if orga_name not in list_of_orga:
                response = post_orga_to_database(orga_name, description)
                if response is not None and response.status_code in (201, 204):
                    fetch_organizations_from_database.clear()
                    st.success(f"id {orga_name} a été créé avec status: 'ongoing'")
                    orga_data = fetch_organizations_from_database()
                    st.dataframe(data=orga_data)
                else:
                    status = getattr(response, 'status_code', '(no response)')
                    body = getattr(response, 'text', '(no response body)')
                    st.error(f"Failed to create id: {status} - {body}")
            else:
                st.error("id existe déjà dans la base de données")

        st.subheader("Change status of e-diag")
        if not orga_data:
            st.info("No organizations available to update.")
        else:
            if isinstance(orga_data, list) and len(orga_data) > 0 and isinstance(orga_data[0], dict):
                org_options = [o.get("name") or o.get("id") or str(o) for o in orga_data]
            else:
                org_options = [str(o) for o in orga_data]

            if not org_options:
                st.info("Could not determine organization names from API response.")
            else:
                selected_orga = st.selectbox("Select organization to change status:", org_options)
                new_status = st.selectbox("New status:", ["ongoing", "completed"])

                if st.button("Update status"):
                    resp = update_organization_status(selected_orga, new_status)
                    if resp is None:
                        st.error("No response from server. See logs for details.")
                    elif resp.status_code in (200, 204):
                        fetch_organizations_from_database.clear()
                        st.success(f"Status for '{selected_orga}' updated to '{new_status}'")
                        refreshed = fetch_organizations_from_database()
                        st.dataframe(data=refreshed)
                    else:
                        try:
                            txt = resp.text
                        except Exception:
                            txt = "(no response body)"
                        st.error(f"Failed to update status: {resp.status_code} - {txt}")
