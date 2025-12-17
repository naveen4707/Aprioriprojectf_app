import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Market Basket Insights",
    page_icon="🛒",
    layout="wide"
)

# --- CSS FOR ATTRACTIVE UI ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    h1 {
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_data
def load_rules():
    try:
        df = pd.read_pickle("rules.pkl")
        return df
    except FileNotFoundError:
        return None

rules_df = load_rules()

# --- SIDEBAR FILTERS ---
st.sidebar.title("🎛️ Settings")
st.sidebar.write("Refine your analysis rules.")

if rules_df is not None:
    # Sliders
    min_sup = st.sidebar.slider("Min Support", 0.01, 0.2, 0.03, 0.01)
    min_conf = st.sidebar.slider("Min Confidence", 0.1, 1.0, 0.3, 0.05)
    min_lift = st.sidebar.slider("Min Lift", 1.0, 10.0, 1.0, 0.1)
    
    # Filter the dataframe based on sliders
    filtered_df = rules_df[
        (rules_df['support'] >= min_sup) &
        (rules_df['confidence'] >= min_conf) &
        (rules_df['lift'] >= min_lift)
    ]
else:
    filtered_df = pd.DataFrame()

# --- MAIN PAGE ---
st.title("🛒 Market Basket Analysis Dashboard")
st.markdown("Discover buying patterns and product associations from your sales data.")

if rules_df is None:
    st.error("⚠️ 'rules.pkl' not found! Please run 'generate_model.py' first.")
    st.stop()

if filtered_df.empty:
    st.warning("No rules found with these settings. Try lowering the filters in the sidebar.")
    st.stop()

# --- METRIC CARDS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Rules Found", len(filtered_df))
col2.metric("Avg Confidence", f"{filtered_df['confidence'].mean():.2%}")
col3.metric("Avg Lift", f"{filtered_df['lift'].mean():.2f}")
col4.metric("Strongest Link", f"{filtered_df['lift'].max():.2f}x")

st.divider()

# --- INTELLIGENT PRODUCT SEARCH ---
st.subheader("🔍 Product Recommender")
all_items = sorted(list(set(filtered_df['antecedents'].unique())))
selected_item = st.selectbox("Select a product to see what customers buy with it:", ["(Select Item)"] + all_items)

if selected_item != "(Select Item)":
    # Find specific rules
    specific_rules = filtered_df[filtered_df['antecedents'] == selected_item].sort_values(by="confidence", ascending=False)
    
    if not specific_rules.empty:
        st.success(f"Customers who buy **{selected_item}** also buy:")
        
        # Display as a clean list
        for _, row in specific_rules.head(5).iterrows():
            st.write(f"👉 **{row['consequents']}** (Confidence: {row['confidence']:.0%}, Lift: {row['lift']:.2f})")
    else:
        st.info(f"No strong associations found for {selected_item} with current settings.")

st.divider()

# --- VISUALIZATION & TABLE TABS ---
tab1, tab2 = st.tabs(["🕸️ Network Graph", "📄 Detailed Data"])

with tab1:
    st.subheader("Association Network")
    st.caption("Visualizing the top 20 strongest rules (by Lift)")
    
    # Graph Logic
    try:
        # Take top 20 rules to prevent messy graph
        graph_data = filtered_df.sort_values(by="lift", ascending=False).head(20)
        
        G = nx.DiGraph()
        for _, row in graph_data.iterrows():
            G.add_edge(row['antecedents'], row['consequents'], weight=row['lift'])
            
        fig, ax = plt.subplots(figsize=(10, 6))
        pos = nx.spring_layout(G, k=0.8)
        
        # Draw nodes and edges
        nx.draw(G, pos, with_labels=True, node_color='skyblue', 
                node_size=2500, font_size=9, font_weight='bold', 
                edge_color='grey', alpha=0.7, arrowsize=15)
        
        # Draw edge labels (Lift values)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        formatted_labels = {k: f"{v:.1f}" for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=formatted_labels, font_size=8)
        
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating graph: {e}")

with tab2:
    st.subheader("Raw Association Rules")
    # Display dataframe with formatting
    st.dataframe(
        filtered_df[['antecedents', 'consequents', 'support', 'confidence', 'lift']],
        use_container_width=True
    )
