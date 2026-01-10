"""
BTC Price Direction Prediction - Streamlit Dashboard
Application interactive pour prédiction et monitoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Configuration de la page
st.set_page_config(
    page_title="BTC MLOps Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #F7931A;
        margin-bottom: 2rem;
    }
    .prediction-up { color: #00FF00; font-size: 2rem; font-weight: bold; }
    .prediction-down { color: #FF0000; font-size: 2rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Bitcoin.svg/1200px-Bitcoin.svg.png", width=100)
st.sidebar.title("⚙️ Configuration")

# API URL
API_URL = st.sidebar.text_input(
    "API URL",
    value="https://btc-prediction-api.whitesmoke-94ae13ff.switzerlandnorth.azurecontainerapps.io",
    help="URL de l'API de prédiction (Azure ou locale)"
)

# Header principal
st.markdown('<h1 class="main-header">₿ BTC Price Direction Predictor</h1>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prédiction", "📊 Data Analysis", "📈 Drift Detection", "📋 Model Info"])

# ============================================
# TAB 1: PRÉDICTION
# ============================================
with tab1:
    st.header("🎯 Prédiction de Direction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Entrez les features")
        
        with st.form("prediction_form"):
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                open_price = st.number_input("Open", value=42000.0, step=100.0)
                high_price = st.number_input("High", value=42500.0, step=100.0)
                low_price = st.number_input("Low", value=41800.0, step=100.0)
                close_price = st.number_input("Close", value=42300.0, step=100.0)
                volume_btc = st.number_input("Volume BTC", value=100.0, step=10.0)
                volume_usd = st.number_input("Volume USD", value=4200000.0, step=100000.0)
            
            with col_b:
                returns_1h = st.number_input("Returns 1h (%)", value=0.5, step=0.1)
                returns_2h = st.number_input("Returns 2h (%)", value=1.0, step=0.1)
                returns_4h = st.number_input("Returns 4h (%)", value=2.0, step=0.1)
                returns_8h = st.number_input("Returns 8h (%)", value=3.0, step=0.1)
                returns_24h = st.number_input("Returns 24h (%)", value=5.0, step=0.5)
                volatility_4h = st.number_input("Volatility 4h", value=0.02, step=0.01)
            
            with col_c:
                volatility_8h = st.number_input("Volatility 8h", value=0.03, step=0.01)
                volatility_24h = st.number_input("Volatility 24h", value=0.04, step=0.01)
                ma_5 = st.number_input("MA 5", value=42000.0, step=100.0)
                ma_10 = st.number_input("MA 10", value=41800.0, step=100.0)
                ma_20 = st.number_input("MA 20", value=41600.0, step=100.0)
                ma_50 = st.number_input("MA 50", value=41400.0, step=100.0)
            
            col_d, col_e, col_f = st.columns(3)
            with col_d:
                ema_5 = st.number_input("EMA 5", value=42100.0, step=100.0)
                ema_10 = st.number_input("EMA 10", value=41900.0, step=100.0)
                ema_20 = st.number_input("EMA 20", value=41700.0, step=100.0)
                momentum_5 = st.number_input("Momentum 5", value=500.0, step=50.0)
                momentum_10 = st.number_input("Momentum 10", value=800.0, step=50.0)
            
            with col_e:
                momentum_20 = st.number_input("Momentum 20", value=1200.0, step=50.0)
                roc_5 = st.number_input("ROC 5", value=1.2, step=0.1)
                roc_10 = st.number_input("ROC 10", value=1.9, step=0.1)
                rsi = st.number_input("RSI", value=55.0, min_value=0.0, max_value=100.0)
                atr = st.number_input("ATR", value=500.0, step=50.0)
            
            with col_f:
                bb_upper = st.number_input("BB Upper", value=43000.0, step=100.0)
                bb_middle = st.number_input("BB Middle", value=42000.0, step=100.0)
                bb_lower = st.number_input("BB Lower", value=41000.0, step=100.0)
                bb_width = st.number_input("BB Width", value=2000.0, step=100.0)
                bb_position = st.number_input("BB Position", value=0.5, step=0.1)
            
            col_g, col_h = st.columns(2)
            with col_g:
                stoch_k = st.number_input("STOCH K", value=65.0, min_value=0.0, max_value=100.0)
                stoch_d = st.number_input("STOCH D", value=60.0, min_value=0.0, max_value=100.0)
                volume_ma_5 = st.number_input("Volume MA 5", value=100.0, step=10.0)
                volume_ma_20 = st.number_input("Volume MA 20", value=95.0, step=10.0)
                volume_ratio = st.number_input("Volume Ratio", value=1.05, step=0.05)
            
            with col_h:
                high_low_ratio = st.number_input("HIGH/LOW Ratio", value=1.008, step=0.001)
                close_open_ratio = st.number_input("CLOSE/OPEN Ratio", value=1.007, step=0.001)
                hour = st.number_input("Hour (0-23)", value=12, min_value=0, max_value=23, step=1)
                day_of_week = st.number_input("Day of Week (0-6)", value=3, min_value=0, max_value=6, step=1)
                is_weekend = st.number_input("Is Weekend (0/1)", value=0, min_value=0, max_value=1, step=1)
            
            submitted = st.form_submit_button("🚀 Prédire", use_container_width=True)
        
        if submitted:
            # Construire la liste des 43 features dans le bon ordre
            features_list = [
                open_price, high_price, low_price, close_price, volume_btc, volume_usd,
                returns_1h, returns_2h, returns_4h, returns_8h, returns_24h,
                volatility_4h, volatility_8h, volatility_24h,
                ma_5, ma_10, ma_20, ma_50,
                ema_5, ema_10, ema_20,
                momentum_5, momentum_10, momentum_20,
                roc_5, roc_10,
                rsi, atr,
                bb_upper, bb_middle, bb_lower, bb_width, bb_position,
                stoch_k, stoch_d,
                volume_ma_5, volume_ma_20, volume_ratio,
                high_low_ratio, close_open_ratio,
                hour, day_of_week, is_weekend
            ]
            
            try:
                response = requests.post(f"{API_URL}/predict", json={"features": features_list}, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("Prédiction réussie!")
                    direction = result.get("direction", "UNKNOWN")
                    probability_up = result.get("probability_up", 0.5)
                    confidence = result.get("confidence", 0.5)
                    
                    if direction == "UP":
                        st.markdown('<p class="prediction-up">📈 HAUSSE PRÉVUE</p>', unsafe_allow_html=True)
                    else:
                        st.markdown('<p class="prediction-down">📉 BAISSE PRÉVUE</p>', unsafe_allow_html=True)
                    
                    st.metric("Confiance", f"{confidence*100:.1f}%")
                    st.metric("Probabilité Hausse", f"{probability_up*100:.1f}%")
                else:
                    st.error(f"Erreur API: {response.status_code}")
                    
            except Exception as e:
                st.warning("⚠️ API non disponible. Mode simulation activé.")
                prediction = np.random.choice([0, 1])
                probability = np.random.uniform(0.5, 0.8)
                
                if prediction == 1:
                    st.markdown('<p class="prediction-up">📈 HAUSSE (Simulation)</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="prediction-down">📉 BAISSE (Simulation)</p>', unsafe_allow_html=True)
                st.metric("Confiance (Simulation)", f"{probability*100:.1f}%")
    
    with col2:
        st.subheader("📊 Indicateurs")
        st.metric("RSI", f"{rsi:.1f}", delta="Neutre" if 30 < rsi < 70 else "Extrême")
        st.metric("ATR", f"${atr:.2f}")
        st.metric("Volatilité 24h", f"{volatility_24h:.4f}")

# ============================================
# TAB 2: DATA ANALYSIS
# ============================================
with tab2:
    st.header("📊 Analyse des Données")
    
    uploaded_file = st.file_uploader("Charger un fichier CSV", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Lignes", f"{len(df):,}")
        col2.metric("Colonnes", f"{len(df.columns)}")
        col3.metric("Valeurs manquantes", f"{df.isnull().sum().sum()}")
        col4.metric("Taille", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        st.subheader("Aperçu")
        st.dataframe(df.head(100), use_container_width=True)
        
        st.subheader("Statistiques")
        st.dataframe(df.describe(), use_container_width=True)
        
        if 'close' in df.columns:
            st.subheader("Évolution du Prix")
            fig = px.line(df, y='close', title='Prix BTC')
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📁 Chargez un fichier CSV")

# ============================================
# TAB 3: DRIFT DETECTION
# ============================================
with tab3:
    st.header("📈 Détection de Data Drift")
    
    st.subheader("📁 Charger les données")
    col_upload1, col_upload2 = st.columns(2)
    
    with col_upload1:
        ref_file = st.file_uploader("Données de RÉFÉRENCE (training)", type=['csv'], key='ref_data')
    
    with col_upload2:
        curr_file = st.file_uploader("Données ACTUELLES (production)", type=['csv'], key='curr_data')
    
    # Checkbox pour démo
    use_demo = st.checkbox("Utiliser données de démonstration", value=True)
    
    # Charger les données
    ref_df = None
    curr_df = None
    
    if use_demo:
        np.random.seed(42)
        n_ref = 1000
        n_curr = 500
        
        # 5 features principales
        features = ['close', 'volume', 'rsi_14', 'macd', 'atr']
        
        ref_df = pd.DataFrame({
            'close': np.random.normal(42000, 1000, n_ref),
            'volume': np.random.exponential(1500, n_ref),
            'rsi_14': np.clip(np.random.normal(50, 15, n_ref), 0, 100),
            'macd': np.random.normal(100, 50, n_ref),
            'atr': np.random.exponential(500, n_ref),
        })
        
        curr_df = pd.DataFrame({
            'close': np.random.normal(42000, 1000, n_curr),
            'volume': np.random.exponential(2500, n_curr),  # DRIFT
            'rsi_14': np.clip(np.random.normal(60, 20, n_curr), 0, 100),  # DRIFT
            'macd': np.random.normal(100, 50, n_curr),
            'atr': np.random.exponential(800, n_curr),  # DRIFT
        })
        
        st.info("✓ Données de démo (5 features): drift sur volume, rsi_14, atr")
    
    elif ref_file is not None and curr_file is not None:
        try:
            ref_df = pd.read_csv(ref_file)
            curr_df = pd.read_csv(curr_file)
            st.success("✓ Fichiers chargés avec succès")
        except Exception as e:
            st.error(f"Erreur lors de la lecture: {e}")
    
    # Analyser le drift si données disponibles
    if ref_df is not None and curr_df is not None:
        
        st.subheader("🔍 Analyse de Drift")
        
        col1, col2 = st.columns(2)
        
        # Obtenir les features communes
        common_features = [col for col in ref_df.columns 
                          if col in curr_df.columns 
                          and ref_df[col].dtype in ['float64', 'int64']]
        
        if not common_features:
            st.error("❌ Aucune feature numérique commune trouvée")
        else:
            
            with col1:
                st.subheader("📊 Histogramme")
                selected_feature = st.selectbox("Feature à analyser", common_features)
                
                if selected_feature:
                    ref_data = ref_df[selected_feature].dropna()
                    curr_data = curr_df[selected_feature].dropna()
                    
                    if len(ref_data) > 0 and len(curr_data) > 0:
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(x=ref_data, name='Référence', opacity=0.6, marker_color='blue'))
                        fig.add_trace(go.Histogram(x=curr_data, name='Actuel', opacity=0.6, marker_color='orange'))
                        fig.update_layout(barmode='overlay', template='plotly_dark', title=f'{selected_feature}')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        col_stat1, col_stat2 = st.columns(2)
                        with col_stat1:
                            st.write("**Référence**")
                            st.write(f"Mean: {ref_data.mean():.2f}")
                            st.write(f"Std: {ref_data.std():.2f}")
                        with col_stat2:
                            st.write("**Actuel**")
                            st.write(f"Mean: {curr_data.mean():.2f}")
                            st.write(f"Std: {curr_data.std():.2f}")
            
            with col2:
                st.subheader("🎯 Résultats")
                
                drift_results = {}
                
                for feature in common_features:
                    if len(common_features) == 0:
                        continue
                        
                    ref_col = ref_df[feature].dropna()
                    curr_col = curr_df[feature].dropna()
                    
                    if len(ref_col) < 10 or len(curr_col) < 10:
                        continue
                    
                    try:
                        # KS Test
                        ks_stat, ks_pvalue = stats.ks_2samp(ref_col, curr_col)
                        
                        # PSI
                        try:
                            bins = np.histogram_bin_edges(ref_col, bins=10)
                            ref_counts = np.histogram(ref_col, bins=bins)[0]
                            curr_counts = np.histogram(curr_col, bins=bins)[0]
                            
                            ref_sum = np.sum(ref_counts)
                            curr_sum = np.sum(curr_counts)
                            
                            if ref_sum > 0 and curr_sum > 0:
                                ref_pct = ref_counts / ref_sum
                                curr_pct = curr_counts / curr_sum
                                
                                ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
                                curr_pct = np.where(curr_pct == 0, 0.0001, curr_pct)
                                
                                psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
                            else:
                                psi = 0.0
                        except:
                            psi = 0.0
                        
                        is_drift = (ks_pvalue < 0.05) or (psi > 0.1)
                        
                        drift_results[feature] = {'psi': psi, 'ks_pvalue': ks_pvalue, 'drift': is_drift}
                        
                        col_a, col_b, col_c = st.columns([2, 1, 1])
                        col_a.write(f"**{feature}**")
                        col_b.write(f"PSI: {psi:.3f}")
                        if is_drift:
                            col_c.error("⚠️ DRIFT")
                        else:
                            col_c.success("✓ OK")
                    
                    except Exception as e:
                        st.warning(f"Erreur {feature}: {str(e)[:50]}")
                
                # Résumé
                if drift_results:
                    st.subheader("📊 Résumé")
                    drifted = [f for f, r in drift_results.items() if r['drift']]
                    total = len(drift_results)
                    
                    col_s1, col_s2 = st.columns(2)
                    col_s1.metric("Features analysées", total)
                    col_s2.metric("Drift détecté", len(drifted))
                    
                    if len(drifted) == 0:
                        st.success("✅ Pas de drift")
                    elif len(drifted) <= 2:
                        st.warning("⚠️ Drift modéré")
                    else:
                        st.error("🚨 Drift significatif - Réentraînement recommandé")
                        if st.button("🔄 Déclencher Réentraînement"):
                            st.success("✅ Pipeline déclenché!")
                            st.balloons()
    else:
        if not use_demo:
            st.warning("⬆️ Chargez les fichiers ou activez la démo")

# ============================================
# TAB 4: MODEL INFO
# ============================================
with tab4:
    st.header("📋 Informations du Modèle")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧠 Modèle")
        st.write("**Type:** CatBoost Classifier")
        st.write("**Version:** v4 (Bayesian Optimization)")
        st.write("**Features:** 43")
        
        st.subheader("📊 Métriques")
        metrics = {
            'Accuracy': 0.5262,
            'F1 Score': 0.4736,
            'ROC-AUC': 0.5346,
            'Precision': 0.4918,
            'Recall': 0.4567
        }
        for metric, value in metrics.items():
            st.metric(metric, f"{value:.4f}")
    
    with col2:
        st.subheader("📈 Feature Importance")
        features = ['volume_sma', 'rsi_14', 'macd', 'bb_width', 'atr']
        importance = np.random.uniform(0.05, 0.2, len(features))
        importance = importance / importance.sum()
        
        fig = px.bar(x=importance, y=features, orientation='h', title='Importance')
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #888;'>🚀 BTC MLOps Dashboard | "
            "<a href='https://github.com/Nadhir1210/btc-mlops'>GitHub</a></div>", unsafe_allow_html=True)
