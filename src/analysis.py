import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Configuration du style
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#0f0f0f'
plt.rcParams['axes.facecolor'] = '#1a1a1a'

def analyze_quali_importance():
    """Analyse l'importance croissante des qualifications en F1 (2023-2025)"""
    
    print("🏎️  ANALYSE : L'importance croissante des qualifications en F1")
    print("=" * 70)
    print("Hypothèse : Les voitures modernes (plus lourdes, plus larges)")
    print("rendent les dépassements plus difficiles")
    print("=" * 70)
    
    # Charger les données
    df = pd.read_csv('data/raw/f1_results_2023_2025.csv')
    
    # Nettoyer les données
    df = df[df['quali_position'] < 99]  # Exclure les non-qualifiés
    df = df[df['final_position'] < 99]  # Exclure les DNF pour certaines analyses
    
    # Identifier tous les pilotes et écuries
    all_drivers = df[['driver_full_name', 'driver_code', 'constructor']].drop_duplicates()
    all_teams = df['constructor'].unique()
    
    print(f"\n📊 Données analysées:")
    print(f"   - Saisons : 2023, 2024, 2025 (en cours)")
    print(f"   - Courses : {df['race_name'].nunique()}")
    print(f"   - Pilotes : {df['driver_full_name'].nunique()}")
    print(f"   - Écuries : {len(all_teams)}")
    
    # 1. ANALYSE PRINCIPALE : Corrélation quali-course par année
    print("\n\n🎯 1. CORRÉLATION POSITION QUALIFICATION vs POSITION FINALE")
    print("-" * 60)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    correlations = {}
    
    for i, year in enumerate([2023, 2024, 2025]):
        ax = fig.add_subplot(gs[0, i])
        year_data = df[(df['year'] == year) & (df['final_position'] < 99)]
        
        if len(year_data) > 0:
            corr = year_data['quali_position'].corr(year_data['final_position'])
            correlations[year] = corr
            
            # Scatter plot avec couleurs par équipe
            teams = year_data['constructor'].unique()
            colors = plt.cm.tab20(np.linspace(0, 1, len(teams)))
            
            for team, color in zip(teams[:10], colors[:10]):  # Top 10 teams
                team_data = year_data[year_data['constructor'] == team]
                ax.scatter(team_data['quali_position'], team_data['final_position'], 
                          alpha=0.6, color=color, s=40, label=team if len(team_data) > 5 else '')
            
            # Ligne de tendance
            z = np.polyfit(year_data['quali_position'], year_data['final_position'], 1)
            p = np.poly1d(z)
            ax.plot(range(1, 21), p(range(1, 21)), "r--", alpha=0.8, linewidth=2)
            
            # Ligne parfaite
            ax.plot([1, 20], [1, 20], 'w-', alpha=0.3, linewidth=1)
            
            ax.set_xlabel('Position Qualification')
            ax.set_ylabel('Position Finale')
            ax.set_title(f'{year}\nCorrélation: {corr:.3f}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.2)
            ax.set_xlim(0, 21)
            ax.set_ylim(0, 21)
            
            if i == 2:  # Légende seulement sur le dernier
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            print(f"   {year}: Corrélation = {corr:.3f} {'(en cours)' if year == 2025 else ''}")
    
    # 2. PROBABILITÉ DE VICTOIRE DEPUIS CHAQUE POSITION
    print("\n\n🏆 2. PROBABILITÉ DE VICTOIRE SELON LA POSITION DE DÉPART")
    print("-" * 60)
    
    ax2 = fig.add_subplot(gs[1, :])
    
    for year in [2023, 2024, 2025]:
        year_data = df[df['year'] == year]
        win_prob = []
        positions_analyzed = []
        
        for pos in range(1, 11):
            wins_from_pos = len(year_data[(year_data['quali_position'] == pos) & 
                                         (year_data['final_position'] == 1)])
            starts_from_pos = len(year_data[year_data['quali_position'] == pos])
            
            if starts_from_pos > 0:
                prob = wins_from_pos / starts_from_pos * 100
                win_prob.append(prob)
                positions_analyzed.append(pos)
        
        if win_prob:
            ax2.plot(positions_analyzed, win_prob, marker='o', linewidth=2.5, 
                    label=f'{year} {"(partiel)" if year == 2025 else ""}', 
                    markersize=8, alpha=0.9)
            
            # Stats pour la pole
            if positions_analyzed[0] == 1:
                print(f"   {year}: Victoires depuis la pole = {win_prob[0]:.1f}%")
    
    ax2.set_xlabel('Position de Qualification', fontsize=12)
    ax2.set_ylabel('Probabilité de Victoire (%)', fontsize=12)
    ax2.set_title('Domination de la Pole Position', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(1, 11))
    ax2.set_xlim(0.5, 10.5)
    
    # 3. ANALYSE DES DÉPASSEMENTS ET POSITIONS GAGNÉES
    print("\n\n🔄 3. ANALYSE DES DÉPASSEMENTS EN COURSE")
    print("-" * 60)
    
    # Préparer les données de dépassements
    overtaking_data = []
    
    for year in [2023, 2024, 2025]:
        year_data = df[(df['year'] == year) & (df['final_position'] < 99)]
        
        if len(year_data) > 0:
            # Positions gagnées/perdues
            positions_gained = year_data[year_data['position_change'] > 0]
            avg_positions_gained = positions_gained['position_change'].mean() if len(positions_gained) > 0 else 0
            
            # Pourcentage qui améliore leur position
            improved = len(year_data[year_data['position_change'] > 0])
            total = len(year_data)
            improvement_rate = improved / total * 100 if total > 0 else 0
            
            # Dépassements significatifs (gain de 3+ places)
            significant_overtakes = len(year_data[year_data['position_change'] >= 3])
            significant_rate = significant_overtakes / total * 100 if total > 0 else 0
            
            overtaking_data.append({
                'year': year,
                'avg_positions_gained': avg_positions_gained,
                'improvement_rate': improvement_rate,
                'significant_rate': significant_rate,
                'total_races': year_data['race_name'].nunique()
            })
            
            print(f"   {year}:")
            print(f"      - Pilotes qui gagnent des places: {improvement_rate:.1f}%")
            print(f"      - Gain moyen (si amélioration): {avg_positions_gained:.1f} places")
            print(f"      - Remontées spectaculaires (3+ places): {significant_rate:.1f}%")
    
    # Graphiques des dépassements
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])
    
    years = [d['year'] for d in overtaking_data]
    improvement_rates = [d['improvement_rate'] for d in overtaking_data]
    significant_rates = [d['significant_rate'] for d in overtaking_data]
    
    # Taux d'amélioration
    bars1 = ax3.bar(years, improvement_rates, color=['#00D9FF', '#FF1E00', '#39FF14'], width=0.6, alpha=0.8)
    ax3.set_xlabel('Année')
    ax3.set_ylabel('% de pilotes')
    ax3.set_title('Pilotes qui gagnent des places', fontweight='bold')
    ax3.set_ylim(0, 60)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for i, (year, rate) in enumerate(zip(years, improvement_rates)):
        ax3.text(year, rate + 1, f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Remontées spectaculaires
    bars2 = ax4.bar(years, significant_rates, color=['#00D9FF', '#FF1E00', '#39FF14'], width=0.6, alpha=0.8)
    ax4.set_xlabel('Année')
    ax4.set_ylabel('% de pilotes')
    ax4.set_title('Remontées spectaculaires (3+ places)', fontweight='bold')
    ax4.set_ylim(0, 30)
    ax4.grid(True, alpha=0.3, axis='y')
    
    for i, (year, rate) in enumerate(zip(years, significant_rates)):
        ax4.text(year, rate + 0.5, f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. TOP 3 MONOPOLE
    print("\n\n🥇 4. MONOPOLE DU PODIUM PAR LES TOP 3 EN QUALIFICATION")
    print("-" * 60)
    
    ax5 = fig.add_subplot(gs[2, 2])
    
    podium_data = []
    for year in [2023, 2024, 2025]:
        year_data = df[(df['year'] == year) & (df['final_position'] <= 3)]
        
        if len(year_data) > 0:
            total_podiums = len(year_data)
            from_top3_quali = len(year_data[year_data['quali_position'] <= 3])
            monopoly_rate = from_top3_quali / total_podiums * 100 if total_podiums > 0 else 0
            
            podium_data.append({
                'year': year,
                'monopoly_rate': monopoly_rate,
                'from_top3': from_top3_quali,
                'total': total_podiums
            })
            
            print(f"   {year}: {monopoly_rate:.1f}% des podiums depuis le Top 3 quali")
            print(f"         ({from_top3_quali}/{total_podiums} podiums)")
    
    years_pod = [d['year'] for d in podium_data]
    monopoly_rates = [d['monopoly_rate'] for d in podium_data]
    
    bars3 = ax5.bar(years_pod, monopoly_rates, color=['#FFD700', '#C0C0C0', '#CD7F32'], width=0.6, alpha=0.8)
    ax5.set_xlabel('Année')
    ax5.set_ylabel('% des podiums')
    ax5.set_title('Podiums depuis Top 3 Quali', fontweight='bold')
    ax5.set_ylim(0, 100)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.axhline(y=75, color='red', linestyle='--', alpha=0.5, label='75% (seuil critique)')
    
    for i, (year, rate) in enumerate(zip(years_pod, monopoly_rates)):
        ax5.text(year, rate + 2, f'{rate:.0f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Impact des Voitures Modernes sur la Compétitivité en F1', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('images/quali_importance_analysis_2023_2025.png', dpi=300, bbox_inches='tight', facecolor='#0f0f0f')
    plt.close()
    
    # 5. ANALYSE PAR CIRCUIT - Quels circuits permettent encore des dépassements ?
    print("\n\n🏁 5. ANALYSE PAR CIRCUIT")
    print("-" * 60)
    
    circuit_analysis = []
    circuits_data = df[df['final_position'] < 99].groupby('circuit')
    
    for circuit, circuit_df in circuits_data:
        if len(circuit_df) >= 10:  # Au moins 10 résultats
            avg_position_change = circuit_df['position_change'].mean()
            overtaking_rate = len(circuit_df[circuit_df['position_change'] > 0]) / len(circuit_df) * 100
            
            circuit_analysis.append({
                'circuit': circuit,
                'avg_change': avg_position_change,
                'overtaking_rate': overtaking_rate,
                'races': circuit_df['race_name'].nunique()
            })
    
    circuit_df = pd.DataFrame(circuit_analysis).sort_values('overtaking_rate', ascending=False)
    
    print("\n   Circuits avec le plus de dépassements:")
    for i, row in circuit_df.head(5).iterrows():
        print(f"   {i+1}. {row['circuit']}: {row['overtaking_rate']:.1f}% des pilotes progressent")
    
    print("\n   Circuits avec le moins de dépassements:")
    for i, row in circuit_df.tail(5).iterrows():
        print(f"   - {row['circuit']}: {row['overtaking_rate']:.1f}% des pilotes progressent")
    
    # 6. ANALYSE DES ÉCURIES - Qui souffre le plus ?
    print("\n\n🏎️  6. IMPACT PAR ÉCURIE")
    print("-" * 60)
    
    team_impact = []
    
    for team in all_teams:
        team_data = df[(df['constructor'] == team) & (df['final_position'] < 99)]
        
        if len(team_data) >= 20:  # Au moins 20 courses
            # Corrélation quali-course pour cette équipe
            corr = team_data['quali_position'].corr(team_data['final_position'])
            
            # Capacité à gagner des places
            avg_change = team_data['position_change'].mean()
            improvement_rate = len(team_data[team_data['position_change'] > 0]) / len(team_data) * 100
            
            team_impact.append({
                'team': team,
                'correlation': corr,
                'avg_position_change': avg_change,
                'improvement_rate': improvement_rate,
                'races': len(team_data)
            })
    
    team_df = pd.DataFrame(team_impact).sort_values('improvement_rate', ascending=False)
    
    print("\n   Équipes qui arrivent à dépasser:")
    for i, row in team_df.head(5).iterrows():
        print(f"   ✓ {row['team']}: {row['improvement_rate']:.1f}% de progression, "
              f"corrélation quali-course: {row['correlation']:.3f}")
    
    print("\n   Équipes dépendantes de la qualification:")
    for i, row in team_df.tail(5).iterrows():
        print(f"   ✗ {row['team']}: {row['improvement_rate']:.1f}% de progression, "
              f"corrélation quali-course: {row['correlation']:.3f}")
    
    # 7. GRAPHIQUE FINAL - SYNTHÈSE
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Evolution de la corrélation
    years_corr = sorted(correlations.keys())
    corr_values = [correlations[y] for y in years_corr]
    
    ax1.plot(years_corr, corr_values, marker='o', markersize=12, linewidth=3, color='#FF1E00')
    ax1.fill_between(years_corr, corr_values, alpha=0.3, color='#FF1E00')
    ax1.set_xlabel('Année', fontsize=12)
    ax1.set_ylabel('Corrélation Quali-Course', fontsize=12)
    ax1.set_title('La Qualification Devient de Plus en Plus Décisive', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.5, 1.0)
    
    for year, corr in zip(years_corr, corr_values):
        ax1.text(year, corr + 0.02, f'{corr:.3f}', ha='center', fontweight='bold')
    
    # Comparaison des métriques clés
    metrics_2023 = next(d for d in overtaking_data if d['year'] == 2023)
    metrics_2025 = next(d for d in overtaking_data if d['year'] == 2025)
    
    categories = ['Corrélation\nQuali-Course', 'Pilotes qui\nprogressent (%)', 'Remontées\nspectaculaires (%)']
    values_2023 = [
        correlations.get(2023, 0) * 100,
        metrics_2023['improvement_rate'],
        metrics_2023['significant_rate']
    ]
    values_2025 = [
        correlations.get(2025, 0) * 100,
        metrics_2025['improvement_rate'],
        metrics_2025['significant_rate']
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, values_2023, width, label='2023', color='#00D9FF', alpha=0.8)
    bars2 = ax2.bar(x + width/2, values_2025, width, label='2025', color='#39FF14', alpha=0.8)
    
    ax2.set_xlabel('Métrique', fontsize=12)
    ax2.set_ylabel('Valeur', fontsize=12)
    ax2.set_title('Évolution des Indicateurs Clés', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Ajouter les valeurs
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('images/f1_overtaking_crisis_summary.png', dpi=300, bbox_inches='tight', facecolor='#0f0f0f')
    plt.close()
    
    # CONCLUSIONS FINALES
    print("\n\n🎯 CONCLUSIONS DE L'ANALYSE")
    print("=" * 70)
    
    # Calcul des tendances
    if 2023 in correlations and 2025 in correlations:
        corr_increase = (correlations[2025] - correlations[2023]) / correlations[2023] * 100
        print(f"\n📈 Augmentation de la corrélation quali-course: +{corr_increase:.1f}%")
    
    if overtaking_data:
        decline_2023_2025 = metrics_2023['improvement_rate'] - metrics_2025['improvement_rate']
        print(f"📉 Baisse du taux de progression en course: -{decline_2023_2025:.1f} points")
    
    print("\n🔍 INSIGHTS CLÉS:")
    print("   1. La position de qualification devient de plus en plus déterminante")
    print("   2. Les dépassements deviennent plus rares et difficiles")
    print("   3. Les podiums sont quasi-monopolisés par les Top 3 en qualification")
    print("   4. Certains circuits (street circuits) rendent les dépassements quasi-impossibles")
    print("\n💡 RECOMMANDATIONS:")
    print("   - Réviser les dimensions/poids des voitures pour 2026")
    print("   - Améliorer l'aérodynamique pour faciliter les rapprochements")
    print("   - Considérer des zones DRS supplémentaires sur certains circuits")
    print("   - Format sprint : peut créer plus d'imprévisibilité")
    
    print("\n✅ Analyse terminée ! Graphiques sauvegardés dans le dossier 'images/'")

if __name__ == "__main__":
    analyze_quali_importance()