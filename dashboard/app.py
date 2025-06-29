import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="F1 Analytics - L'importance des Qualifications",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #1a1a1a;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #333;
    }
    h1 {
        color: #FF1E00;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.title("🏎️ F1 Analytics : L'Importance Croissante des Qualifications")
st.markdown("### Les voitures modernes rendent-elles les dépassements impossibles ?")
st.markdown("---")

# Charger les données
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/raw/f1_results_2023_2025.csv')
        # Nettoyer les données
        df['year'] = df['year'].astype(int)
        df['final_position'] = pd.to_numeric(df['final_position'], errors='coerce')
        df['quali_position'] = pd.to_numeric(df['quali_position'], errors='coerce')
        df['position_change'] = pd.to_numeric(df['position_change'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("Aucune donnée disponible. Veuillez d'abord exécuter le script de collecte de données.")
    st.stop()

# Sidebar pour les filtres
st.sidebar.header("🎯 Filtres")

# Sélection de l'année
selected_years = st.sidebar.multiselect(
    "Sélectionner les années",
    options=sorted(df['year'].unique()),
    default=sorted(df['year'].unique())
)

# Sélection des équipes
all_teams = sorted(df['constructor'].unique())
selected_teams = st.sidebar.multiselect(
    "Sélectionner les équipes",
    options=all_teams,
    default=all_teams[:10] if len(all_teams) > 10 else all_teams
)

# Filtrer les données
df_filtered = df[
    (df['year'].isin(selected_years)) & 
    (df['constructor'].isin(selected_teams))
]

# Métriques principales
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_races = df_filtered['race_name'].nunique()
    st.metric("Courses analysées", total_races)

with col2:
    avg_correlation = df_filtered.groupby('year').apply(
        lambda x: x['quali_position'].corr(x['final_position'])
    ).mean()
    st.metric("Corrélation Quali-Course", f"{avg_correlation:.3f}")

with col3:
    improvement_rate = (df_filtered['position_change'] > 0).mean() * 100
    st.metric("Taux de progression", f"{improvement_rate:.1f}%")

with col4:
    podium_from_top3 = df_filtered[
        (df_filtered['final_position'] <= 3) & 
        (df_filtered['quali_position'] <= 3)
    ].shape[0] / df_filtered[df_filtered['final_position'] <= 3].shape[0] * 100
    st.metric("Podiums depuis Top 3", f"{podium_from_top3:.0f}%")

# Tabs pour différentes analyses
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Vue d'ensemble", 
    "🏁 Corrélation Quali-Course", 
    "🔄 Analyse des Dépassements",
    "🏆 Domination de la Pole",
    "📈 Tendances par Équipe"
])

with tab1:
    st.header("Vue d'ensemble de l'analyse")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Evolution de la corrélation par année
        corr_by_year = df_filtered.groupby('year').apply(
            lambda x: x['quali_position'].corr(x['final_position'])
        ).reset_index(name='correlation')
        
        fig_corr = go.Figure()
        fig_corr.add_trace(go.Scatter(
            x=corr_by_year['year'],
            y=corr_by_year['correlation'],
            mode='lines+markers',
            line=dict(color='#FF1E00', width=3),
            marker=dict(size=12),
            fill='tonexty',
            name='Corrélation'
        ))
        
        fig_corr.update_layout(
            title="Évolution de la Corrélation Qualification-Course",
            xaxis_title="Année",
            yaxis_title="Corrélation",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        st.markdown("### 🔍 Insights Clés")
        st.info(
            """
            **Constat principal :**
            La corrélation entre position de qualification et position finale 
            augmente chaque année, indiquant que les dépassements deviennent 
            de plus en plus difficiles.
            
            **Causes possibles :**
            - Voitures plus larges et lourdes
            - Aérodynamique sensible
            - Pneus difficiles à gérer
            - Circuits inadaptés
            """
        )
    
    # Statistiques par circuit
    st.subheader("Performance par Circuit")
    
    circuit_stats = df_filtered.groupby('circuit').agg({
        'position_change': ['mean', 'std'],
        'race_name': 'count'
    }).round(2)
    
    circuit_stats.columns = ['Changement moyen', 'Écart-type', 'Nombre de courses']
    circuit_stats = circuit_stats.sort_values('Changement moyen', ascending=False)
    
    fig_circuits = px.bar(
        circuit_stats.head(10).reset_index(),
        x='circuit',
        y='Changement moyen',
        title="Top 10 Circuits pour les Dépassements",
        template="plotly_dark"
    )
    
    st.plotly_chart(fig_circuits, use_container_width=True)

with tab2:
    st.header("Analyse de la Corrélation Qualification-Course")
    
    # Scatter plot interactif
    fig_scatter = px.scatter(
        df_filtered[df_filtered['final_position'] < 99],
        x='quali_position',
        y='final_position',
        color='constructor',
        facet_col='year',
        trendline="ols",
        title="Position de Qualification vs Position Finale",
        template="plotly_dark",
        hover_data=['driver', 'race_name']
    )
    
    fig_scatter.update_layout(height=600)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Heatmap de corrélation par équipe
    st.subheader("Corrélation par Équipe")
    
    team_correlations = []
    for team in selected_teams:
        team_data = df_filtered[
            (df_filtered['constructor'] == team) & 
            (df_filtered['final_position'] < 99)
        ]
        if len(team_data) > 10:
            corr = team_data['quali_position'].corr(team_data['final_position'])
            team_correlations.append({'team': team, 'correlation': corr})
    
    if team_correlations:
        team_corr_df = pd.DataFrame(team_correlations).sort_values('correlation', ascending=False)
        
        fig_team_corr = px.bar(
            team_corr_df,
            x='team',
            y='correlation',
            title="Dépendance à la Qualification par Équipe",
            template="plotly_dark",
            color='correlation',
            color_continuous_scale='RdYlGn_r'
        )
        
        st.plotly_chart(fig_team_corr, use_container_width=True)

with tab3:
    st.header("Analyse des Dépassements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution des changements de position
        fig_dist = px.histogram(
            df_filtered[df_filtered['final_position'] < 99],
            x='position_change',
            nbins=30,
            title="Distribution des Changements de Position",
            template="plotly_dark",
            color_discrete_sequence=['#00D9FF']
        )
        
        fig_dist.add_vline(x=0, line_dash="dash", line_color="red")
        fig_dist.update_layout(
            xaxis_title="Positions gagnées/perdues",
            yaxis_title="Fréquence"
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # Taux de progression par année
        progression_by_year = df_filtered.groupby('year').apply(
            lambda x: (x['position_change'] > 0).mean() * 100
        ).reset_index(name='progression_rate')
        
        fig_prog = go.Figure()
        fig_prog.add_trace(go.Bar(
            x=progression_by_year['year'],
            y=progression_by_year['progression_rate'],
            marker_color=['#00D9FF', '#FF1E00', '#39FF14'][:len(progression_by_year)]
        ))
        
        fig_prog.update_layout(
            title="Taux de Progression par Année",
            xaxis_title="Année",
            yaxis_title="% de pilotes qui progressent",
            template="plotly_dark"
        )
        
        st.plotly_chart(fig_prog, use_container_width=True)
    
    # Meilleurs dépasseurs
    st.subheader("🏆 Meilleurs Dépasseurs")
    
    overtakers = df_filtered[df_filtered['position_change'] > 0].groupby('driver').agg({
        'position_change': ['sum', 'mean', 'count']
    }).round(2)
    
    overtakers.columns = ['Total positions gagnées', 'Moyenne par course', 'Nombre de courses']
    overtakers = overtakers.sort_values('Total positions gagnées', ascending=False).head(10)
    
    st.dataframe(overtakers, use_container_width=True)

with tab4:
    st.header("Domination de la Pole Position")
    
    # Probabilité de victoire par position de départ
    win_prob_data = []
    
    for pos in range(1, 11):
        for year in selected_years:
            year_data = df_filtered[df_filtered['year'] == year]
            wins = len(year_data[
                (year_data['quali_position'] == pos) & 
                (year_data['final_position'] == 1)
            ])
            starts = len(year_data[year_data['quali_position'] == pos])
            
            if starts > 0:
                win_prob_data.append({
                    'position': pos,
                    'year': year,
                    'win_probability': wins / starts * 100,
                    'starts': starts
                })
    
    if win_prob_data:
        win_prob_df = pd.DataFrame(win_prob_data)
        
        fig_win_prob = px.line(
            win_prob_df,
            x='position',
            y='win_probability',
            color='year',
            title="Probabilité de Victoire selon la Position de Qualification",
            template="plotly_dark",
            markers=True
        )
        
        fig_win_prob.update_layout(
            xaxis_title="Position de Qualification",
            xaxis=dict(dtick=1),
            yaxis_title="Probabilité de Victoire (%)"
        )
        
        st.plotly_chart(fig_win_prob, use_container_width=True)
    
    # Podiums depuis différentes positions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pole_to_win = df_filtered[
            (df_filtered['quali_position'] == 1) & 
            (df_filtered['final_position'] == 1)
        ].shape[0] / df_filtered[df_filtered['quali_position'] == 1].shape[0] * 100
        
        st.metric("Pole → Victoire", f"{pole_to_win:.1f}%")
    
    with col2:
        top3_to_podium = df_filtered[
            (df_filtered['quali_position'] <= 3) & 
            (df_filtered['final_position'] <= 3)
        ].shape[0] / df_filtered[df_filtered['quali_position'] <= 3].shape[0] * 100
        
        st.metric("Top 3 → Podium", f"{top3_to_podium:.1f}%")
    
    with col3:
        outside_top10_to_points = df_filtered[
            (df_filtered['quali_position'] > 10) & 
            (df_filtered['final_position'] <= 10)
        ].shape[0]
        
        st.metric("Hors Top 10 → Points", outside_top10_to_points)

with tab5:
    st.header("Tendances par Équipe")
    
    # Sélection d'une équipe spécifique
    selected_team = st.selectbox(
        "Sélectionner une équipe pour l'analyse détaillée",
        options=selected_teams
    )
    
    team_data = df_filtered[df_filtered['constructor'] == selected_team]
    
    if not team_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance des pilotes de l'équipe
            driver_stats = team_data.groupby('driver').agg({
                'points': 'sum',
                'position_change': 'mean',
                'final_position': 'mean'
            }).round(2)
            
            driver_stats.columns = ['Points totaux', 'Changement moyen', 'Position moyenne']
            driver_stats = driver_stats.sort_values('Points totaux', ascending=False)
            
            st.subheader(f"Performance des pilotes - {selected_team}")
            st.dataframe(driver_stats, use_container_width=True)
        
        with col2:
            # Evolution de la performance
            team_evolution = team_data.groupby(['year', 'round']).agg({
                'points': 'sum'
            }).reset_index()
            
            team_evolution['cumulative_points'] = team_evolution.groupby('year')['points'].cumsum()
            
            fig_team_evo = px.line(
                team_evolution,
                x='round',
                y='cumulative_points',
                color='year',
                title=f"Évolution des points - {selected_team}",
                template="plotly_dark",
                markers=True
            )
            
            st.plotly_chart(fig_team_evo, use_container_width=True)
        
        # Comparaison quali vs course
        st.subheader("Comparaison Qualification vs Course")
        
        avg_quali = team_data.groupby('driver')['quali_position'].mean()
        avg_race = team_data.groupby('driver')['final_position'].mean()
        
        comparison_df = pd.DataFrame({
            'driver': avg_quali.index,
            'quali_avg': avg_quali.values,
            'race_avg': avg_race.values
        })
        
        comparison_df['difference'] = comparison_df['quali_avg'] - comparison_df['race_avg']
        
        fig_comparison = go.Figure()
        
        fig_comparison.add_trace(go.Bar(
            name='Position Moyenne Qualification',
            x=comparison_df['driver'],
            y=comparison_df['quali_avg'],
            marker_color='lightblue'
        ))
        
        fig_comparison.add_trace(go.Bar(
            name='Position Moyenne Course',
            x=comparison_df['driver'],
            y=comparison_df['race_avg'],
            marker_color='lightcoral'
        ))
        
        fig_comparison.update_layout(
            title=f"Positions moyennes - {selected_team}",
            barmode='group',
            template="plotly_dark",
            yaxis=dict(autorange='reversed'),
            xaxis_title="Pilote",
            yaxis_title="Position"
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)

# Footer avec conclusions
st.markdown("---")
st.markdown("### 🎯 Conclusions de l'Analyse")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **📈 Tendances Observées:**
    - Corrélation quali-course en hausse
    - Moins de dépassements chaque année
    - Domination accrue de la pole position
    """)

with col2:
    st.markdown("""
    **🔧 Causes Techniques:**
    - Voitures plus larges (2m vs 1.8m)
    - Poids accru (+150kg depuis 2014)
    - Aérodynamique sensible au dirty air
    """)

with col3:
    st.markdown("""
    **💡 Solutions Proposées:**
    - Révision du règlement 2026
    - Voitures plus légères et étroites
    - Amélioration du suivi en course
    """)

# Informations supplémentaires
with st.expander("ℹ️ À propos de cette analyse"):
    st.markdown("""
    Cette analyse examine l'évolution de la Formule 1 entre 2023 et 2025, 
    en se concentrant sur l'importance croissante des qualifications dans 
    le résultat final des courses.
    
    **Sources de données:**
    - API Ergast F1
    - Données simulées pour les courses futures
    
    **Méthodologie:**
    - Analyse de corrélation entre positions de qualification et d'arrivée
    - Étude des changements de position en course
    - Comparaison entre équipes et circuits
    
    **Dernière mise à jour:** {pd.Timestamp.now().strftime('%Y-%m-%d')}
    """)

# Bouton d'export
if st.button("📊 Exporter les données filtrées"):
    csv = df_filtered.to_csv(index=False)
    st.download_button(
        label="Télécharger CSV",
        data=csv,
        file_name=f"f1_data_filtered_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )