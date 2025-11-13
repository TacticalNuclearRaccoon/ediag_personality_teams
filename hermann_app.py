import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from itertools import combinations

DATABASE_URL = st.secrets["DATABASE_URL"]
DATABASE_API_KEY = st.secrets["DATABASE_API_KEY"]
icon = "Favicon_2x.ico"
st.set_page_config(layout='wide', page_icon=icon, page_title="Outil d’analyse Bousole des personnalités")

try:
    st.image("Banniere argios.png", use_container_width=True)
except:
    st.image("Banniere argios.png", use_column_width=True)

st.title("Outil d'analyse pour le e-diagnostic des personnalités")


def fetch_results_from_database():
    url = f"{DATABASE_URL}/rest/v1/hermann_teams"
    headers = {
        "apikey": DATABASE_API_KEY,
        "Authorization": f"Bearer {DATABASE_API_KEY}",
        "Content-Type": "application/json",
    }
    params = {
        "select": "id,user,organisation,A_score,B_score,C_score,D_score, evaluation"
    }
    response = requests.get(url, headers=headers, params=params)
    print(response.status_code, response.text)
    response.raise_for_status()
    return response.json()

### Functions for compatibility graphs
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
def visualize_compatibility_network_colored(compatibility_matrix, threshold=1):
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
    pos = polygon_layout(G)

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
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', edgecolors='black', node_size=1000)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors)
    nx.draw_networkx_labels(G, pos, font_weight='bold', font_size=12)

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
def visualize_person_centered_network_colored(compatibility_matrix, center_person, threshold=1):
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
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_color=['lightblue' if n != center_person else 'gold' for n in G.nodes()], edgecolors='black', node_size=1000)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors)
    nx.draw_networkx_labels(G, pos, font_weight='bold', font_size=12)

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
    compatibility_matrix = pd.DataFrame(0.0, index=df.index, columns=df.index)

    for person1 in df.index:
        for person2 in df.index:
            if person1 == person2:
                compatibility_matrix.loc[person1, person2] = 100.0
                continue

            row1 = df.loc[person1]
            row2 = df.loc[person2]

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
    compatibility_matrix = pd.DataFrame(0.0, index=df.index, columns=df.index)

    for person1 in df.index:
        for person2 in df.index:
            if person1 == person2:
                compatibility_matrix.loc[person1, person2] = 5.0
                continue

            row1 = df.loc[person1]
            row2 = df.loc[person2]

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

data = fetch_results_from_database()

organisations = [row["organisation"] for row in data]
unique_organisations = list(set(organisations))
selected_organisation = st.selectbox("Select organisation", unique_organisations)
df=pd.DataFrame([row for row in data if row["organisation"] == selected_organisation])
df.rename(columns={"user": "Person", "organisation": "Organisation", "A_score": "A", "B_score": "B", "C_score": "C", "D_score": "D"}, inplace=True)
# Create a unique label combining person name and ID to handle duplicate names
df["PersonLabel"] = df.apply(lambda row: f"{row['Person']} (ID: {row['id']})", axis=1)
# Set the PersonLabel as index (which is now unique)
df = df.set_index("PersonLabel")
# Keep the original person name and ID as columns for reference
df['PersonName'] = df['Person']
df['PersonID'] = df['id']
#df["Dominant"] = df.idxmax(axis=1)
#st.subheader("🧪 Raw Data Preview")
#st.dataframe(data=df)
# Sidebar for user interaction

selected_view = st.sidebar.radio("Choisir une analyse:", ["Profil individuel", "Profil moyen de l'équipe", "Styles dominants", "Déviations", "Les autres", "Compatibilité", "Communication"])


# Radar chart helper
def radar_chart(name, scores, color='lightblue'):
    # Create figure
    fig = go.Figure()

    # Add the data trace
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=['A', 'B', 'C', 'D'],
        fill='toself',
        name=name,
        line=dict(color=color, width=2)
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
        showlegend=False,
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

# View 1: Individual radar chart
if selected_view == "Profil individuel":
    selected_person = st.selectbox("Sélectionner un membre de l'équipe:", df.index.tolist())
    if selected_person:
        person_scores = df.loc[selected_person][["A", "B", "C", "D"]].tolist()
        fig = radar_chart(selected_person, person_scores)
        st.plotly_chart(fig)
    else:
        st.warning("Veuillez sélectionner un membre de l'équipe.")


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
    st.write(df['Dominant'])
    
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
    # Heatmap
    df_scores = df.drop(columns=["PersonLabel", "Organisation", "evaluation"])
    #st.dataframe(data=df_scores)

    # Standardize (Z-score) so deviations stand out
    df_norm = (df_scores - df_scores.mean()) / df_scores.std()

    melted = df[["A", "B", "C", "D"]].melt(var_name="Quadrant", value_name="Score")
    # this same data can be used in the boxplot
    quadrant_data = [
        melted[melted['Quadrant'] == 'A']['Score'].values,
        melted[melted['Quadrant'] == 'B']['Score'].values,
        melted[melted['Quadrant'] == 'C']['Score'].values,
        melted[melted['Quadrant'] == 'D']['Score'].values
    ]

    quadrant_labels = ['A', 'B', 'C', 'D']

    
    plt.figure(figsize=(3, 3))
    fig_diverge, ax_diverge = plt.subplots(figsize=(5,5))
    sns.heatmap(df_norm, annot=df_scores, cmap="coolwarm", center=0, cbar_kws={'label': 'Z-score'}, fmt='.1f')
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
    fig, axs = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    for i, cat in enumerate(categories):
        axs[i].hlines(y=df_scores.index, xmin=avg_scores[cat], xmax=df_scores[cat], color='grey', alpha=0.5)
        axs[i].plot(df_scores[cat], df_scores.index, "o", label="Score")
        axs[i].vlines(avg_scores[cat], 0, len(df_scores), color="red", linestyles="dashed", label="Moyenne")
        axs[i].set_title(f"Quadrant {cat}")
        axs[i].set_xlabel("Score")
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
    #st.write(minim_comp)
    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        st.subheader("Réseau de compatibilité")
        fig = visualize_compatibility_network_colored(similarity_matrix)
        st.pyplot(fig)
        st.markdown("""
* Chaque nœud représente un membre de l'équipe.
* Les arêtes représentent la compatibilité entre les membres.
* La largeur des arêtes est proportionnelle au score de compatibilité.
* Les couleurs des arêtes vont du rouge (faible compatibilité) au vert (forte compatibilité).
* Le score de compatibilité est indiqué par le texte sur les arêtes""")
    with comp_col2:
        st.subheader("Matrice de compatibilité")
        plt.figure(figsize=(10, 10))
        fig_comp, ax_comp = plt.subplots(figsize=(8,8))
        sns.heatmap(similarity_matrix, annot=True, cmap='BuPu', fmt=".1f")
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
    team_matrix = new_matrix[new_matrix['Person'].isin(best_team)]
    #st.dataframe(data=team_matrix)
    team_matrix.set_index('Person', inplace=True)
    fig_team = visualize_compatibility_network_colored(team_matrix)
    st.pyplot(fig_team)


elif selected_view == "Les autres":
    selected_person = st.selectbox("Sélectionner un membre de l'équipe:", df.index.tolist())
    if selected_person:
        st.header(f"L'équipe selon {selected_person}")

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
                names = set()
                for key in person_evaluations_dict.keys():
                    parts = key.split('_', 1)
                    if len(parts) == 2:
                        quadrant, name = parts
                        if quadrant in ['A', 'B', 'C', 'D']:
                            names.add(name)

                if not names:
                    st.warning(f"Le dictionnaire d'évaluation pour {selected_person} est vide ou ne contient pas de données exploitables.")
                    st.write(person_evaluations_dict)
                else:
                    for name in sorted(list(names)):
                        col1, col2 = st.columns(2)

                        with col1:
                            scores_by_other = [
                                person_evaluations_dict.get(f'A_{name}', 0) * 25,
                                person_evaluations_dict.get(f'B_{name}', 0) * 25,
                                person_evaluations_dict.get(f'C_{name}', 0) * 25,
                                person_evaluations_dict.get(f'D_{name}', 0) * 25
                            ]
                            fig_by_other = radar_chart(name, scores_by_other, color='mediumpurple')
                            title_by_other = f"Profil de {name}<br>(vu par {selected_person})"
                            fig_by_other.update_layout(title=dict(text=title_by_other, font=dict(size=16)))
                            st.plotly_chart(fig_by_other, use_container_width=True)

                        with col2:
                            if name in df.index:
                                real_scores = df.loc[name, ["A", "B", "C", "D"]].tolist()
                                fig_real = radar_chart(name, real_scores, color='lightblue')
                                title_real = f"Profil de {name}<br>(auto-évaluation)"
                                fig_real.update_layout(title=dict(text=title_real, font=dict(size=16)))
                                st.plotly_chart(fig_real, use_container_width=True)
                            else:
                                st.warning(f"Le profil de {name} n'a pas pu être trouvé.")
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
    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        st.subheader("Réseau de compatibilité")
        fig = visualize_person_centered_network_colored(similarity_matrix, comm_person)
        st.pyplot(fig)
        st.markdown("""
* Chaque nœud représente un membre de l'équipe.
* Les arêtes représentent la compatibilité entre les membres.
* La largeur des arêtes est proportionnelle au score de compatibilité.
* Les couleurs des arêtes vont du rouge (faible compatibilité) au vert (forte compatibilité).
* Le score de compatibilité est indiqué par le texte sur les arêtes""")
    with comp_col2:
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
            