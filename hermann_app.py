import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json

DATABASE_URL = st.secrets["DATABASE_URL"]
DATABASE_API_KEY = st.secrets["DATABASE_API_KEY"]
icon = "Favicon_2x.ico"
st.set_page_config(layout='wide', page_icon=icon, page_title="Outil d’analyse Bousole des personnalités")


st.title("Outil d'analyze pour le e-diagnostic des personnalités")


def fetch_results_from_database():
    url = f"{DATABASE_URL}/rest/v1/hermann_teams"
    headers = {
        "apikey": DATABASE_API_KEY,
        "Authorization": f"Bearer {DATABASE_API_KEY}",
        "Content-Type": "application/json",
    }
    params = {
        "select": "user,organisation,A_score,B_score,C_score,D_score, evaluation"
    }
    response = requests.get(url, headers=headers, params=params)
    print(response.status_code, response.text)
    response.raise_for_status()
    return response.json()

data = fetch_results_from_database()

organisations = [row["organisation"] for row in data]
unique_organisations = list(set(organisations))
selected_organisation = st.selectbox("Select organisation", unique_organisations)
df=pd.DataFrame([row for row in data if row["organisation"] == selected_organisation])
df.rename(columns={"user": "Person", "organisation": "Organisation", "A_score": "A", "B_score": "B", "C_score": "C", "D_score": "D"}, inplace=True)
# there can be 2 people with the same within the same organisation
df["PersonLabel"] = df.apply(lambda row: f"{row.name}: {row['Person']} ({row['Organisation']})", axis=1)
df = df.set_index("Person")
#df["Dominant"] = df.idxmax(axis=1)
#st.subheader("🧪 Raw Data Preview")
#st.dataframe(data=df)
# Sidebar for user interaction

selected_view = st.sidebar.radio("Choisir une analyse:", ["Profil individuel", "Profil moyen de l'équipe", "Styles dominants", "Déviations", "Les autres"])


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
    st.write("Calculated dominant quadrant per person:")
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
    ax.set_title("Distribution of Dominant Quadrants", color='white', fontsize=12)
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig)
    with col2:
        st.dataframe(dominant_counts_df_display)

# View 5: Déviations
elif selected_view == "Déviations":
    # Heatmap
    df_scores = df.drop(columns=["PersonLabel", "Organisation"])

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
        st.subheader("Scores moyennes par quandrant")
        meanz = pd.DataFrame(df[["A", "B", "C", "D"]].mean(), columns=["Valeur moyenne dans l'équipe"])
        st.dataframe(data=meanz)
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
            st.write("Aucune valeur divergeante détectée")


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
        axs[i].set_title(f"Quandrant {cat}")
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