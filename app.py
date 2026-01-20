import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="MMM Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        font-weight: bold;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stDataFrame {
        border: 2px solid #1f77b4;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = False
if 'media_data' not in st.session_state:
    st.session_state.media_data = {}
if 'kpi_data' not in st.session_state:
    st.session_state.kpi_data = None
if 'combined_data' not in st.session_state:
    st.session_state.combined_data = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'promotion_data' not in st.session_state:
    st.session_state.promotion_data = None

# Helper functions for MMM
def adstock_transformation(x, alpha=0.5):
    """Apply adstock (geometric decay) transformation"""
    y = np.zeros_like(x, dtype=float)
    if len(x) > 0:
        y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = x[t] + alpha * y[t-1]
    return y

def hill_transformation(x, kappa, slope=1.0):
    """Apply Hill saturation transformation"""
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    k = max(float(kappa), 1e-9)
    if slope == 1.0:
        return x / (x + k)
    xs = np.power(x, slope)
    ks = np.power(k, slope)
    return xs / (xs + ks)

def hill_derivative(x, kappa, slope=1.0):
    """Calculate derivative of Hill function for marginal ROAS"""
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    k = max(float(kappa), 1e-9)
    if slope == 1.0:
        return k / (x + k)**2
    xs = np.power(x, slope)
    ks = np.power(k, slope)
    return slope * np.power(x, slope - 1.0) * ks / (xs + ks)**2

def calculate_metrics(y_true, y_pred):
    """Calculate model performance metrics"""
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) if mask.sum() > 0 else 0
    
    # wMAPE
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) if np.sum(np.abs(y_true)) > 0 else 0
    
    return r2, mape, wmape

def add_seasonality_features(df, date_col):
    """Add seasonality features: day of week and month"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Day of week (0=Monday, 6=Sunday)
    df['day_of_week'] = df[date_col].dt.dayofweek
    
    # Month of year (1-12)
    df['month'] = df[date_col].dt.month
    
    # Create dummy variables for day of week and month
    day_dummies = pd.get_dummies(df['day_of_week'], prefix='dow', drop_first=True)
    month_dummies = pd.get_dummies(df['month'], prefix='month', drop_first=True)
    
    # Combine
    df_with_seasonality = pd.concat([df, day_dummies, month_dummies], axis=1)
    
    return df_with_seasonality

def process_promotion_variable(df, promo_col):
    """
    Process promotion variable - convert to dummy if string, use as numeric if numeric
    Returns: tuple (processed_df, promotion_feature_cols, is_dummy)
    """
    df = df.copy()
    
    # Check if column is string/object type
    if df[promo_col].dtype == 'object' or df[promo_col].dtype.name == 'category':
        # String values - convert to dummies
        promo_dummies = pd.get_dummies(df[promo_col], prefix='promo', drop_first=True)
        df = pd.concat([df, promo_dummies], axis=1)
        feature_cols = promo_dummies.columns.tolist()
        is_dummy = True
    else:
        # Numeric values - use as is
        feature_cols = [promo_col]
        is_dummy = False
    
    return df, feature_cols, is_dummy

# Main app
st.markdown('<p class="main-header">📊 Marketing Mix Modeling Platform</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/analytics.png", width=100)
    st.markdown("### Navigation")
    tab_selection = st.radio(
        "Select a section:",
        ["📤 Data Upload", "🔍 Data Overview", "🎯 Marketing Mix Modeling", "📈 Results & Insights"],
        key="navigation"
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    This platform helps you:
    - Upload media spend & KPI data
    - Add promotion/discount variables
    - Analyze marketing effectiveness
    - Optimize budget allocation (scipy)
    - Generate actionable insights
    """)

# TAB 1: Data Upload
if tab_selection == "📤 Data Upload":
    st.markdown('<p class="sub-header">Upload Your Marketing Data</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 KPI Data (Revenue)")
        st.info("Upload your store/Shopify revenue data. Must include: **Date** and **Revenue** columns")
        
        kpi_file = st.file_uploader(
            "Choose KPI CSV file",
            type=['csv'],
            key='kpi_upload',
            help="Upload CSV with Date and Revenue columns"
        )
        
        if kpi_file:
            try:
                kpi_df = pd.read_csv(kpi_file)
                st.session_state.kpi_data = kpi_df
                
                st.success(f"✅ KPI data uploaded successfully! ({len(kpi_df)} rows)")
                
                with st.expander("Preview KPI Data"):
                    st.dataframe(kpi_df.head(10), use_container_width=True)
                    
                    # Show basic stats
                    st.markdown("**Data Info:**")
                    st.write(f"- Columns: {', '.join(kpi_df.columns.tolist())}")
                    st.write(f"- Date range: {kpi_df.iloc[:, 0].min()} to {kpi_df.iloc[:, 0].max()}")
                    
            except Exception as e:
                st.error(f"Error loading KPI data: {str(e)}")
    
    with col2:
        st.markdown("#### 💰 Media Spend Data")
        st.info("Upload media channel data. Must include: **Date** and **Cost** columns (optional: Clicks, Impressions)")
        
        num_channels = st.number_input("Number of media channels", min_value=1, max_value=10, value=2, key='num_channels')
        
        for i in range(num_channels):
            st.markdown(f"**Channel {i+1}:**")
            channel_name = st.text_input(f"Channel name", value=f"Channel_{i+1}", key=f'channel_name_{i}')
            channel_file = st.file_uploader(
                f"Upload {channel_name} CSV",
                type=['csv'],
                key=f'channel_file_{i}'
            )
            
            if channel_file:
                try:
                    channel_df = pd.read_csv(channel_file)
                    st.session_state.media_data[channel_name] = channel_df
                    st.success(f"✅ {channel_name} uploaded ({len(channel_df)} rows)")
                    
                    with st.expander(f"Preview {channel_name}"):
                        st.dataframe(channel_df.head(5), use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error loading {channel_name}: {str(e)}")
    
    # Promotion/Discount variable upload
    st.markdown("---")
    st.markdown("#### 🎁 Promotion/Discount Data (Optional)")
    st.info("""
    Upload promotion data with **Date** and **Promotion** columns.
    - **String values** (e.g., 'Yes'/'No', 'Sale'/'Normal') → Converted to dummy variables
    - **Numeric values** (e.g., 10%, 0.15) → Used as continuous variable
    """)
    
    promo_file = st.file_uploader(
        "Upload Promotion CSV (optional)",
        type=['csv'],
        key='promo_upload',
        help="CSV with Date and Promotion columns"
    )
    
    if promo_file:
        try:
            promo_df = pd.read_csv(promo_file)
            st.session_state.promotion_data = promo_df
            
            st.success(f"✅ Promotion data uploaded! ({len(promo_df)} rows)")
            
            with st.expander("Preview Promotion Data"):
                st.dataframe(promo_df.head(10), use_container_width=True)
                
                # Detect type
                promo_col = promo_df.columns[1]
                if promo_df[promo_col].dtype == 'object':
                    st.info(f"✓ Detected **categorical** promotion: {promo_df[promo_col].unique()}")
                else:
                    st.info(f"✓ Detected **numeric** promotion: Range {promo_df[promo_col].min():.2f} - {promo_df[promo_col].max():.2f}")
                    
        except Exception as e:
            st.error(f"Error loading promotion data: {str(e)}")
    
    # Combine data button
    st.markdown("---")
    if st.button("🔗 Combine All Data", type="primary", use_container_width=True):
        if st.session_state.kpi_data is None:
            st.error("❌ Please upload KPI data first!")
        elif len(st.session_state.media_data) == 0:
            st.error("❌ Please upload at least one media channel!")
        else:
            with st.spinner("Combining data..."):
                try:
                    # Start with KPI data
                    combined = st.session_state.kpi_data.copy()
                    date_col = combined.columns[0]
                    combined[date_col] = pd.to_datetime(combined[date_col])
                    
                    # Merge each media channel
                    for channel_name, channel_df in st.session_state.media_data.items():
                        channel_df = channel_df.copy()
                        channel_date_col = channel_df.columns[0]
                        channel_df[channel_date_col] = pd.to_datetime(channel_df[channel_date_col])
                        
                        # Rename columns to include channel name
                        rename_dict = {}
                        for col in channel_df.columns:
                            if col.lower() not in ['date']:
                                rename_dict[col] = f"{channel_name}_{col}"
                        channel_df = channel_df.rename(columns=rename_dict)
                        
                        # Merge
                        channel_df = channel_df.rename(columns={channel_date_col: date_col})
                        combined = combined.merge(channel_df, on=date_col, how='left')
                    
                    # Merge promotion data if available
                    if st.session_state.promotion_data is not None:
                        promo_df = st.session_state.promotion_data.copy()
                        promo_date_col = promo_df.columns[0]
                        promo_df[promo_date_col] = pd.to_datetime(promo_df[promo_date_col])
                        promo_df = promo_df.rename(columns={promo_date_col: date_col})
                        combined = combined.merge(promo_df, on=date_col, how='left')
                        
                        # Fill missing promotion values
                        promo_col = promo_df.columns[1]
                        if combined[promo_col].dtype == 'object':
                            combined[promo_col] = combined[promo_col].fillna('None')
                        else:
                            combined[promo_col] = combined[promo_col].fillna(0)
                    
                    # Fill NaN with 0 for cost columns
                    cost_cols = [col for col in combined.columns if 'cost' in col.lower() or 'spend' in col.lower()]
                    combined[cost_cols] = combined[cost_cols].fillna(0)
                    
                    st.session_state.combined_data = combined
                    st.session_state.data_uploaded = True
                    
                    st.success("✅ Data combined successfully!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Error combining data: {str(e)}")

# TAB 2: Data Overview
elif tab_selection == "🔍 Data Overview":
    st.markdown('<p class="sub-header">Data Overview & Validation</p>', unsafe_allow_html=True)
    
    if not st.session_state.data_uploaded:
        st.warning("⚠️ Please upload and combine data first in the 'Data Upload' tab!")
    else:
        df = st.session_state.combined_data
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Data summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Records", len(df))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            date_range_days = (df[date_col].max() - df[date_col].min()).days
            st.metric("Date Range (Days)", date_range_days)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            date_range_months = date_range_days / 30
            st.metric("Months of Data", f"{date_range_months:.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            media_channels = len([col for col in df.columns if 'cost' in col.lower() or 'spend' in col.lower()])
            st.metric("Media Channels", media_channels)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Data validation
        st.markdown("---")
        st.markdown("### ✅ Data Validation")
        
        validation_col1, validation_col2, validation_col3 = st.columns(3)
        
        with validation_col1:
            # Check for minimum 24 months
            if date_range_months >= 24:
                st.success(f"✅ Sufficient data: {date_range_months:.1f} months (≥24 months required)")
            else:
                st.error(f"❌ Insufficient data: {date_range_months:.1f} months (<24 months)")
        
        with validation_col2:
            # Check for required columns
            has_revenue = any('revenue' in col.lower() for col in df.columns)
            if has_revenue:
                st.success("✅ Revenue column found")
            else:
                st.error("❌ Revenue column not found")
        
        with validation_col3:
            # Check for promotion data
            has_promo = any('promo' in col.lower() or 'discount' in col.lower() for col in df.columns)
            if has_promo:
                st.success("✅ Promotion data included")
            else:
                st.info("ℹ️ No promotion data")
        
        # Display combined data
        st.markdown("---")
        st.markdown("### 📊 Combined Dataset")
        
        # Styled dataframe
        st.dataframe(
            df.style.background_gradient(subset=[col for col in df.columns if col != date_col], cmap='Blues'),
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Combined Data",
            data=csv,
            file_name=f"combined_mmm_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Basic statistics
        st.markdown("---")
        st.markdown("### 📈 Descriptive Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Numerical Summary:**")
            st.dataframe(df.describe(), use_container_width=True)
        
        with col2:
            st.markdown("**Missing Values:**")
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Count': df.isnull().sum().values,
                'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
            })
            st.dataframe(missing_df, use_container_width=True)
        
        # Correlation heatmap
        st.markdown("---")
        st.markdown("### 🔥 Correlation Heatmap")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(12, 8))
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
            plt.title('Correlation Matrix of Media Channels and KPI', fontsize=16, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)

# TAB 3: Marketing Mix Modeling
elif tab_selection == "🎯 Marketing Mix Modeling":
    st.markdown('<p class="sub-header">Marketing Mix Modeling</p>', unsafe_allow_html=True)
    
    if not st.session_state.data_uploaded:
        st.warning("⚠️ Please upload and combine data first in the 'Data Upload' tab!")
        st.info("👉 Go to the 'Data Upload' tab to upload your media and KPI data.")
    else:
        df = st.session_state.combined_data.copy()
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Check data sufficiency
        date_range_months = (df[date_col].max() - df[date_col].min()).days / 30
        
        if date_range_months < 24:
            st.error(f"❌ Insufficient data for modeling: {date_range_months:.1f} months available (24 months required)")
            st.stop()
        
        st.success(f"✅ Data validation passed: {date_range_months:.1f} months available")
        
        # Column mapping
        st.markdown("---")
        st.markdown("### 🔧 Configure Model Variables")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Target Variable (KPI):**")
            potential_target_cols = [col for col in df.columns if 'revenue' in col.lower() or 'sales' in col.lower()]
            target_col = st.selectbox(
                "Select target/KPI column",
                potential_target_cols if potential_target_cols else df.columns[1:],
                key='target_col_selector'
            )
        
        with col2:
            st.markdown("**Date Column:**")
            date_col_confirm = st.selectbox(
                "Confirm date column",
                [date_col],
                key='date_col_confirm'
            )
        
        st.markdown("**Media Spend Columns:**")
        cost_cols = [col for col in df.columns if 'cost' in col.lower() or 'spend' in col.lower()]
        media_cols = st.multiselect(
            "Select media spend columns",
            cost_cols if cost_cols else df.columns[1:],
            default=cost_cols if cost_cols else [],
            key='media_cols_selector'
        )
        
        if not media_cols:
            st.warning("⚠️ Please select at least one media spend column!")
            st.stop()
        
        # Promotion variable selection
        st.markdown("**Promotion/Discount Variable (Optional):**")
        promo_options = [col for col in df.columns if ('promo' in col.lower() or 'discount' in col.lower()) 
                         and col not in media_cols and col != target_col and col != date_col]
        
        promo_col = None
        if promo_options:
            use_promo = st.checkbox("Include promotion variable", value=True)
            if use_promo:
                promo_col = st.selectbox("Select promotion column", promo_options, key='promo_col_selector')
        
        # Other control variables
        st.markdown("**Other Control Variables (Optional):**")
        available_controls = [col for col in df.columns if col not in media_cols and col != target_col 
                             and col != date_col and col != promo_col 
                             and not ('promo' in col.lower() or 'discount' in col.lower())]
        control_cols = st.multiselect(
            "Select additional control variables",
            available_controls,
            key='other_controls_selector'
        )
        
        # Model parameters
        st.markdown("---")
        st.markdown("### ⚙️ Model Parameters")
        
        param_col1, param_col2, param_col3 = st.columns(3)
        
        with param_col1:
            adstock_alpha = st.slider("Adstock Rate (α)", 0.0, 0.9, 0.5, 0.05, help="Carryover effect of advertising")
        
        with param_col2:
            hill_slope = st.slider("Hill Slope", 0.5, 2.0, 1.0, 0.1, help="Saturation curve shape")
        
        with param_col3:
            train_test_split = st.slider("Train/Test Split", 0.6, 0.9, 0.8, 0.05, help="Proportion of data for training")
        
        # Run model button
        st.markdown("---")
        if st.button("🚀 Run Marketing Mix Model", type="primary", use_container_width=True):
            with st.spinner("Training Marketing Mix Model... This may take a few minutes."):
                try:
                    # Work with daily data
                    st.info("Step 1/7: Preparing daily data...")
                    daily_df = df.copy()
                    daily_df = daily_df.sort_values(date_col).reset_index(drop=True)
                    
                    # Add seasonality
                    st.info("Step 2/7: Adding seasonality features...")
                    daily_df = add_seasonality_features(daily_df, date_col)
                    
                    # Process promotion variable if selected
                    promo_features = []
                    promo_is_dummy = False
                    if promo_col:
                        st.info(f"Step 3/7: Processing promotion variable ({promo_col})...")
                        daily_df, promo_features, promo_is_dummy = process_promotion_variable(daily_df, promo_col)
                        st.session_state.promo_is_dummy = promo_is_dummy
                        st.session_state.promo_features = promo_features
                    else:
                        st.info("Step 3/7: No promotion variable selected, skipping...")
                        st.session_state.promo_features = []
                        st.session_state.promo_is_dummy = False
                    
                    # Engineer media features
                    st.info("Step 4/7: Engineering media features (adstock + saturation)...")
                    
                    meta = {}
                    feat_cols = []
                    
                    for media_col in media_cols:
                        # Adstock
                        daily_df[f'{media_col}_adstock'] = adstock_transformation(
                            daily_df[media_col].values, alpha=adstock_alpha
                        )
                        
                        # Hill saturation
                        kappa = np.nanmedian(daily_df[f'{media_col}_adstock'].values)
                        if not np.isfinite(kappa) or kappa <= 0:
                            kappa = np.nanmean(daily_df[f'{media_col}_adstock'].values) or 1.0
                        
                        daily_df[f'{media_col}_saturated'] = hill_transformation(
                            daily_df[f'{media_col}_adstock'].values,
                            kappa=kappa,
                            slope=hill_slope
                        )
                        
                        # Standardize
                        mu = daily_df[f'{media_col}_saturated'].mean()
                        sd = daily_df[f'{media_col}_saturated'].std() or 1.0
                        
                        feat_name = f'{media_col}_feat'
                        daily_df[feat_name] = (daily_df[f'{media_col}_saturated'] - mu) / sd
                        
                        feat_cols.append(feat_name)
                        
                        # Store metadata
                        meta[feat_name] = {
                            'spend_col': media_col,
                            'kappa': kappa,
                            'slope': hill_slope,
                            'mu': mu,
                            'sd': sd
                        }
                    
                    # Train/test split
                    st.info("Step 5/7: Splitting data into train and test sets...")
                    split_idx = int(len(daily_df) * train_test_split)
                    train_df = daily_df.iloc[:split_idx].copy()
                    test_df = daily_df.iloc[split_idx:].copy()
                    
                    # Prepare X and y
                    seasonality_cols = [col for col in daily_df.columns if 'dow_' in col or 'month_' in col]
                    
                    # Combine all features
                    all_control_cols = control_cols + promo_features
                    
                    X_train = pd.concat([
                        pd.Series(1.0, index=train_df.index, name='const'),
                        train_df[feat_cols],
                        train_df[all_control_cols] if all_control_cols else pd.DataFrame(index=train_df.index),
                        train_df[seasonality_cols]
                    ], axis=1).astype('float64')
                    
                    X_test = pd.concat([
                        pd.Series(1.0, index=test_df.index, name='const'),
                        test_df[feat_cols],
                        test_df[all_control_cols] if all_control_cols else pd.DataFrame(index=test_df.index),
                        test_df[seasonality_cols]
                    ], axis=1).astype('float64')
                    
                    y_train = train_df[target_col].values.astype(float)
                    y_test = test_df[target_col].values.astype(float)
                    
                    # Train model
                    st.info("Step 6/7: Training OLS regression model...")
                    model = sm.OLS(y_train, X_train).fit()
                    
                    # Predictions
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    st.info("Step 7/7: Calculating performance metrics...")
                    train_r2, train_mape, train_wmape = calculate_metrics(y_train, y_train_pred)
                    test_r2, test_mape, test_wmape = calculate_metrics(y_test, y_test_pred)
                    
                    # Store results in session state
                    st.session_state.model_trained = True
                    st.session_state.model = model
                    st.session_state.meta = meta
                    st.session_state.feat_cols = feat_cols
                    st.session_state.media_cols = media_cols
                    st.session_state.target_col = target_col
                    st.session_state.date_col = date_col
                    st.session_state.train_df = train_df
                    st.session_state.test_df = test_df
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.y_train_pred = y_train_pred
                    st.session_state.y_test_pred = y_test_pred
                    st.session_state.adstock_alpha = adstock_alpha
                    st.session_state.promo_col = promo_col
                    st.session_state.control_cols = control_cols
                    
                    st.success("✅ Model trained successfully!")
                    st.balloons()
                    
                    # Display metrics
                    st.markdown("---")
                    st.markdown("### 📊 Model Performance")
                    
                    metric_col1, metric_col2 = st.columns(2)
                    
                    with metric_col1:
                        st.markdown("**Training Set:**")
                        st.metric("R²", f"{train_r2:.3f}")
                        st.metric("MAPE", f"{train_mape:.2%}")
                        st.metric("wMAPE", f"{train_wmape:.2%}")
                    
                    with metric_col2:
                        st.markdown("**Test Set:**")
                        st.metric("R²", f"{test_r2:.3f}")
                        st.metric("MAPE", f"{test_mape:.2%}")
                        st.metric("wMAPE", f"{test_wmape:.2%}")
                    
                    # Model fit plot
                    st.markdown("---")
                    st.markdown("### 📈 Model Fit Visualization")
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    
                    # Train
                    ax1.plot(train_df[date_col], y_train, label='Actual', color='green', alpha=0.7)
                    ax1.plot(train_df[date_col], y_train_pred, label='Predicted', color='blue', alpha=0.7)
                    ax1.set_title(f'Training Set (R²={train_r2:.3f})', fontsize=14, fontweight='bold')
                    ax1.set_xlabel('Date')
                    ax1.set_ylabel(target_col)
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Test
                    ax2.plot(test_df[date_col], y_test, label='Actual', color='green', alpha=0.7)
                    ax2.plot(test_df[date_col], y_test_pred, label='Predicted', color='blue', alpha=0.7)
                    ax2.set_title(f'Test Set (R²={test_r2:.3f})', fontsize=14, fontweight='bold')
                    ax2.set_xlabel('Date')
                    ax2.set_ylabel(target_col)
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.info("✅ Model training complete! Go to 'Results & Insights' tab to view detailed analysis.")
                    
                except Exception as e:
                    st.error(f"❌ Error during modeling: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

# TAB 4: Results & Insights
elif tab_selection == "📈 Results & Insights":
    st.markdown('<p class="sub-header">Results & Insights</p>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first in the 'Marketing Mix Modeling' tab!")
        st.info("👉 Go to the 'Marketing Mix Modeling' tab to configure and train your model.")
    else:
        # Retrieve from session state
        model = st.session_state.model
        meta = st.session_state.meta
        feat_cols = st.session_state.feat_cols
        media_cols = st.session_state.media_cols
        target_col = st.session_state.target_col
        date_col = st.session_state.date_col
        test_df = st.session_state.test_df
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        y_test_pred = st.session_state.y_test_pred
        adstock_alpha = st.session_state.adstock_alpha
        promo_col = st.session_state.promo_col
        promo_features = st.session_state.promo_features
        promo_is_dummy = st.session_state.promo_is_dummy
        control_cols = st.session_state.control_cols
        
        # Tabs for different analyses
        result_tabs = st.tabs([
            "📊 Channel Contribution",
            "💰 ROI Analysis",
            "📈 Response Curves",
            "🎯 Budget Optimization (Scipy)",
            "📋 Model Summary"
        ])
        
        # Tab 1: Channel Contribution
        with result_tabs[0]:
            st.markdown("### Channel Contribution to Revenue")
            
            # Calculate contributions
            contributions = {}
            for feat in feat_cols:
                beta = float(model.params.get(feat, 0.0))
                contrib = np.sum(X_test[feat].values * beta)
                channel_name = meta[feat]['spend_col']
                contributions[channel_name] = contrib
            
            # Add promotion contribution
            if promo_col and promo_features:
                promo_contrib = 0
                for promo_feat in promo_features:
                    if promo_feat in X_test.columns:
                        beta = float(model.params.get(promo_feat, 0.0))
                        promo_contrib += np.sum(X_test[promo_feat].values * beta)
                contributions['Promotion'] = promo_contrib
            
            # Add baseline
            baseline = float(model.params.get('const', 0.0)) * len(X_test)
            contributions['Baseline'] = baseline
            
            contrib_df = pd.DataFrame.from_dict(contributions, orient='index', columns=['Contribution'])
            contrib_df['Contribution %'] = 100 * contrib_df['Contribution'] / contrib_df['Contribution'].sum()
            contrib_df = contrib_df.sort_values('Contribution', ascending=False)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**Contribution Summary:**")
                st.dataframe(
                    contrib_df.style.format({'Contribution': '{:,.0f}', 'Contribution %': '{:.1f}%'}),
                    use_container_width=True
                )
            
            with col2:
                # Pie chart
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = plt.cm.Set3(range(len(contrib_df)))
                ax.pie(contrib_df['Contribution'], labels=contrib_df.index, autopct='%1.1f%%',
                       colors=colors, startangle=90)
                ax.set_title('Revenue Contribution by Channel', fontsize=14, fontweight='bold')
                st.pyplot(fig)
            
            # Bar chart
            st.markdown("---")
            fig, ax = plt.subplots(figsize=(12, 6))
            contrib_df['Contribution'].plot(kind='barh', ax=ax, color='steelblue')
            ax.set_xlabel('Revenue Contribution', fontsize=12)
            ax.set_title('Channel Contribution to Total Revenue', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(contrib_df['Contribution']):
                ax.text(v, i, f' {v:,.0f}', va='center', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Tab 2: ROI Analysis
        with result_tabs[1]:
            st.markdown("### Return on Investment (ROI) Analysis")
            
            roi_data = []
            
            for feat in feat_cols:
                channel_name = meta[feat]['spend_col']
                beta = float(model.params.get(feat, 0.0))
                
                # Total contribution
                contrib = np.sum(X_test[feat].values * beta)
                
                # Total spend
                total_spend = test_df[channel_name].sum()
                
                # ROI (iROAS)
                roi = contrib / total_spend if total_spend > 0 else 0
                
                # Marginal ROI at current spend
                kappa = meta[feat]['kappa']
                slope = meta[feat]['slope']
                sd = meta[feat]['sd']
                
                current_avg_spend = test_df[channel_name].mean()
                A = current_avg_spend / (1 - adstock_alpha)
                
                marginal_roas = (beta / sd) * hill_derivative(A, kappa, slope) / (1 - adstock_alpha)
                
                roi_data.append({
                    'Channel': channel_name.replace('_Cost', '').replace('_cost', ''),
                    'Total Spend': total_spend,
                    'Revenue Contribution': contrib,
                    'ROI (iROAS)': roi,
                    'Marginal ROI': marginal_roas
                })
            
            roi_df = pd.DataFrame(roi_data).sort_values('ROI (iROAS)', ascending=False)
            
            # Display table
            st.dataframe(
                roi_df.style.format({
                    'Total Spend': '{:,.0f}',
                    'Revenue Contribution': '{:,.0f}',
                    'ROI (iROAS)': '{:.2f}',
                    'Marginal ROI': '{:.2f}'
                }).background_gradient(subset=['ROI (iROAS)', 'Marginal ROI'], cmap='RdYlGn'),
                use_container_width=True
            )
            
            # ROI visualization
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                roi_df.plot(x='Channel', y='ROI (iROAS)', kind='bar', ax=ax, color='coral', legend=False)
                ax.set_title('ROI by Channel', fontsize=14, fontweight='bold')
                ax.set_ylabel('ROI (Revenue per $ Spent)', fontsize=12)
                ax.set_xlabel('')
                ax.axhline(y=1, color='red', linestyle='--', label='Break-even')
                ax.legend()
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                roi_df.plot(x='Channel', y='Marginal ROI', kind='bar', ax=ax, color='skyblue', legend=False)
                ax.set_title('Marginal ROI by Channel', fontsize=14, fontweight='bold')
                ax.set_ylabel('Marginal ROI', fontsize=12)
                ax.set_xlabel('')
                ax.axhline(y=1, color='red', linestyle='--', label='Break-even')
                ax.legend()
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
            
            # Insights
            st.markdown("---")
            st.markdown("### 💡 Key Insights")
            
            best_roi_channel = roi_df.iloc[0]
            best_marginal_channel = roi_df.sort_values('Marginal ROI', ascending=False).iloc[0]
            
            insight_col1, insight_col2 = st.columns(2)
            
            with insight_col1:
                st.info(f"""
                **Best Overall ROI:**
                - **{best_roi_channel['Channel']}** has the highest ROI at **{best_roi_channel['ROI (iROAS)']:.2f}**
                - For every $1 spent, you get ${best_roi_channel['ROI (iROAS)']:.2f} in revenue
                """)
            
            with insight_col2:
                st.info(f"""
                **Best Marginal Efficiency:**
                - **{best_marginal_channel['Channel']}** has the highest marginal ROI at **{best_marginal_channel['Marginal ROI']:.2f}**
                - This channel has the most room for additional investment
                """)
        
        # Tab 3: Response Curves
        with result_tabs[2]:
            st.markdown("### Saturation & Response Curves")
            
            selected_channel = st.selectbox(
                "Select channel to analyze",
                [meta[feat]['spend_col'] for feat in feat_cols],
                key='curve_channel'
            )
            
            # Find corresponding feature
            feat = [f for f in feat_cols if meta[f]['spend_col'] == selected_channel][0]
            
            beta = float(model.params.get(feat, 0.0))
            kappa = meta[feat]['kappa']
            slope = meta[feat]['slope']
            mu = meta[feat]['mu']
            sd = meta[feat]['sd']
            
            # Generate spend range
            historical_spend = test_df[selected_channel].values
            max_spend = np.percentile(historical_spend, 95)
            spend_range = np.linspace(0, max_spend * 1.5, 200)
            
            # Calculate responses
            adstocked = spend_range / (1 - adstock_alpha)
            saturated = hill_transformation(adstocked, kappa, slope)
            standardized = (saturated - mu) / sd
            revenue = beta * standardized
            
            # Calculate marginal ROAS
            marginal_roas = (beta / sd) * hill_derivative(adstocked, kappa, slope) / (1 - adstock_alpha)
            
            # Calculate iROAS
            iroas = np.zeros_like(revenue)
            for i in range(1, len(revenue)):
                iroas[i] = revenue[i] / spend_range[i] if spend_range[i] > 0 else 0
            
            # Plotting
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Saturation curve
            axes[0, 0].plot(spend_range, revenue, color='steelblue', linewidth=2)
            axes[0, 0].axvline(historical_spend.mean(), color='red', linestyle='--', label='Current avg spend')
            axes[0, 0].set_title('Saturation Curve', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Daily Spend', fontsize=11)
            axes[0, 0].set_ylabel('Incremental Revenue', fontsize=11)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Marginal ROAS
            axes[0, 1].plot(spend_range, marginal_roas, color='coral', linewidth=2)
            axes[0, 1].axvline(historical_spend.mean(), color='red', linestyle='--', label='Current avg spend')
            axes[0, 1].axhline(y=1, color='green', linestyle='--', label='Break-even')
            axes[0, 1].set_title('Marginal ROAS', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Daily Spend', fontsize=11)
            axes[0, 1].set_ylabel('Marginal ROAS', fontsize=11)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # iROAS curve
            axes[1, 0].plot(spend_range[1:], iroas[1:], color='purple', linewidth=2)
            axes[1, 0].axvline(historical_spend.mean(), color='red', linestyle='--', label='Current avg spend')
            axes[1, 0].axhline(y=1, color='green', linestyle='--', label='Break-even')
            axes[1, 0].set_title('Incremental ROAS', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Daily Spend', fontsize=11)
            axes[1, 0].set_ylabel('iROAS', fontsize=11)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Spend efficiency
            efficiency = revenue / spend_range
            efficiency[0] = 0
            axes[1, 1].plot(spend_range, efficiency, color='green', linewidth=2)
            axes[1, 1].axvline(historical_spend.mean(), color='red', linestyle='--', label='Current avg spend')
            axes[1, 1].set_title('Spend Efficiency', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Daily Spend', fontsize=11)
            axes[1, 1].set_ylabel('Revenue / Spend', fontsize=11)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Current metrics
            st.markdown("---")
            st.markdown("### 📊 Current Performance Metrics")
            
            current_spend = historical_spend.mean()
            current_idx = np.argmin(np.abs(spend_range - current_spend))
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("Current Avg Spend", f"${current_spend:,.0f}")
            
            with metric_col2:
                st.metric("Marginal ROAS", f"{marginal_roas[current_idx]:.2f}")
            
            with metric_col3:
                st.metric("iROAS", f"{iroas[current_idx]:.2f}")
            
            with metric_col4:
                saturation_level = (saturated[current_idx] / saturated[-1]) * 100
                st.metric("Saturation Level", f"{saturation_level:.1f}%")
        
        # Tab 4: Budget Optimization (Scipy)
        with result_tabs[3]:
            st.markdown("### Budget Allocation Optimizer (Scipy SLSQP)")
            
            st.info("""
            **Optimization Method:** Using scipy.optimize.minimize with SLSQP solver
            - Maximizes total revenue subject to budget constraint
            - Accounts for adstock carryover and saturation effects
            - Finds optimal spend allocation across all channels
            """)
            
            # Current budget
            current_budget = sum([test_df[meta[feat]['spend_col']].sum() for feat in feat_cols])
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                new_budget = st.slider(
                    "Total Budget",
                    min_value=int(current_budget * 0.5),
                    max_value=int(current_budget * 2),
                    value=int(current_budget),
                    step=int(current_budget * 0.05),
                    format="$%d"
                )
            
            with col2:
                budget_change = ((new_budget - current_budget) / current_budget) * 100
                st.metric("Budget Change", f"{budget_change:+.1f}%")
            
            # Run optimization button
            if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
                with st.spinner("Running scipy optimization..."):
                    try:
                        # Objective function
                        def mmm_objective(channel_totals):
                            """Calculate negative total revenue (we minimize)"""
                            total_revenue = 0
                            
                            for i, feat in enumerate(feat_cols):
                                channel_name = meta[feat]['spend_col']
                                beta = float(model.params.get(feat, 0.0))
                                kappa = meta[feat]['kappa']
                                slope = meta[feat]['slope']
                                sd = meta[feat]['sd']
                                mu = meta[feat]['mu']
                                
                                # Get optimized channel spend
                                optimized_spend = channel_totals[i]
                                
                                # Calculate average daily spend for test period
                                n_days = len(test_df)
                                avg_daily_spend = optimized_spend / n_days
                                
                                # Apply adstock and saturation
                                adstocked = avg_daily_spend / (1 - adstock_alpha)
                                saturated = hill_transformation(adstocked, kappa, slope)
                                standardized = (saturated - mu) / sd
                                
                                # Calculate contribution
                                channel_revenue = beta * standardized * n_days
                                total_revenue += channel_revenue
                            
                            # Return negative (we're minimizing)
                            return -total_revenue
                        
                        # Budget constraint
                        def budget_constraint(channel_totals):
                            return np.sum(channel_totals) - new_budget
                        
                        # Initial guess - current totals
                        initial_totals = [test_df[meta[feat]['spend_col']].sum() for feat in feat_cols]
                        
                        # Bounds - all >= 0
                        bounds = [(0, None) for _ in feat_cols]
                        
                        # Solve
                        solution = minimize(
                            fun=mmm_objective,
                            x0=initial_totals,
                            bounds=bounds,
                            method="SLSQP",
                            constraints={
                                'type': 'eq',
                                'fun': budget_constraint
                            },
                            options={
                                'maxiter': 1000,
                                'ftol': 1e-9
                            }
                        )
                        
                        if solution.success:
                            st.success("✅ Optimization completed successfully!")
                            
                            # Create results dataframe
                            allocation_data = []
                            for i, feat in enumerate(feat_cols):
                                channel_name = meta[feat]['spend_col']
                                current_spend = test_df[channel_name].sum()
                                optimized_spend = solution.x[i]
                                
                                allocation_data.append({
                                    'Channel': channel_name.replace('_Cost', '').replace('_cost', ''),
                                    'Current Spend': current_spend,
                                    'Optimized Spend': optimized_spend,
                                    'Change': optimized_spend - current_spend,
                                    'Change %': ((optimized_spend - current_spend) / current_spend * 100) if current_spend > 0 else 0
                                })
                            
                            alloc_df = pd.DataFrame(allocation_data)
                            
                            # Display results
                            st.markdown("---")
                            st.markdown("#### 📊 Optimal Budget Allocation")
                            
                            st.dataframe(
                                alloc_df.style.format({
                                    'Current Spend': '{:,.0f}',
                                    'Optimized Spend': '{:,.0f}',
                                    'Change': '{:+,.0f}',
                                    'Change %': '{:+.1f}%'
                                }).background_gradient(subset=['Change %'], cmap='RdYlGn', vmin=-50, vmax=50),
                                use_container_width=True
                            )
                            
                            # Visualization
                            st.markdown("---")
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                            
                            # Current vs Optimized
                            x = np.arange(len(alloc_df))
                            width = 0.35
                            
                            ax1.bar(x - width/2, alloc_df['Current Spend'], width, label='Current', color='steelblue')
                            ax1.bar(x + width/2, alloc_df['Optimized Spend'], width, label='Optimized', color='coral')
                            ax1.set_xlabel('Channel')
                            ax1.set_ylabel('Budget')
                            ax1.set_title('Current vs Optimized Budget (Scipy)', fontsize=14, fontweight='bold')
                            ax1.set_xticks(x)
                            ax1.set_xticklabels(alloc_df['Channel'], rotation=45, ha='right')
                            ax1.legend()
                            ax1.grid(axis='y', alpha=0.3)
                            
                            # Budget change
                            colors = ['green' if x > 0 else 'red' for x in alloc_df['Change %']]
                            ax2.barh(alloc_df['Channel'], alloc_df['Change %'], color=colors)
                            ax2.set_xlabel('Change (%)')
                            ax2.set_title('Budget Change by Channel', fontsize=14, fontweight='bold')
                            ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
                            ax2.grid(axis='x', alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Expected impact
                            st.markdown("---")
                            st.markdown("### 📈 Expected Impact")
                            
                            current_revenue = y_test.sum()
                            optimized_revenue = -solution.fun  # Negative because we minimized
                            expected_lift = optimized_revenue - current_revenue
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Current Revenue", f"${current_revenue:,.0f}")
                            
                            with col2:
                                st.metric("Optimized Revenue", f"${optimized_revenue:,.0f}", delta=f"${expected_lift:,.0f}")
                            
                            with col3:
                                lift_pct = (expected_lift / current_revenue) * 100
                                st.metric("Expected Lift", f"{lift_pct:+.1f}%")
                            
                            # Optimization details
                            with st.expander("🔧 Optimization Details"):
                                st.write(f"**Method:** {solution.message}")
                                st.write(f"**Iterations:** {solution.nit}")
                                st.write(f"**Function Evaluations:** {solution.nfev}")
                                st.write(f"**Objective Value:** {-solution.fun:,.2f}")
                        
                        else:
                            st.error(f"❌ Optimization failed: {solution.message}")
                    
                    except Exception as e:
                        st.error(f"❌ Error during optimization: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        
        # Tab 5: Model Summary
        with result_tabs[4]:
            st.markdown("### Model Summary")
            
            # Model coefficients
            st.markdown("#### 📊 Model Coefficients")
            
            coef_data = []
            for param in model.params.index:
                coef_data.append({
                    'Variable': param,
                    'Coefficient': model.params[param],
                    'Std Error': model.bse[param],
                    'T-Statistic': model.tvalues[param],
                    'P-Value': model.pvalues[param],
                    'Significant': '***' if model.pvalues[param] < 0.001 else ('**' if model.pvalues[param] < 0.01 else ('*' if model.pvalues[param] < 0.05 else ''))
                })
            
            coef_df = pd.DataFrame(coef_data)
            
            st.dataframe(
                coef_df.style.format({
                    'Coefficient': '{:.4f}',
                    'Std Error': '{:.4f}',
                    'T-Statistic': '{:.4f}',
                    'P-Value': '{:.4f}'
                }).background_gradient(subset=['Coefficient'], cmap='coolwarm', vmin=-1, vmax=1),
                use_container_width=True
            )
            
            st.caption("Significance: *** p<0.001, ** p<0.01, * p<0.05")
            
            # Model statistics
            st.markdown("---")
            st.markdown("#### 📈 Model Statistics")
            
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            
            with stat_col1:
                st.metric("R-squared", f"{model.rsquared:.4f}")
                st.metric("Adj. R-squared", f"{model.rsquared_adj:.4f}")
            
            with stat_col2:
                st.metric("F-statistic", f"{model.fvalue:.2f}")
                st.metric("Prob (F-statistic)", f"{model.f_pvalue:.4e}")
            
            with stat_col3:
                st.metric("AIC", f"{model.aic:.2f}")
                st.metric("BIC", f"{model.bic:.2f}")
            
            # Model diagnostics
            st.markdown("---")
            st.markdown("#### 🔍 Model Diagnostics")
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Residuals vs Fitted
            residuals = y_test - y_test_pred
            axes[0, 0].scatter(y_test_pred, residuals, alpha=0.5)
            axes[0, 0].axhline(y=0, color='red', linestyle='--')
            axes[0, 0].set_xlabel('Fitted Values')
            axes[0, 0].set_ylabel('Residuals')
            axes[0, 0].set_title('Residuals vs Fitted', fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Q-Q plot
            stats.probplot(residuals, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('Normal Q-Q Plot', fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Histogram of residuals
            axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
            axes[1, 0].set_xlabel('Residuals')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Distribution of Residuals', fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Actual vs Predicted
            axes[1, 1].scatter(y_test, y_test_pred, alpha=0.5)
            axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                           'r--', lw=2, label='Perfect Prediction')
            axes[1, 1].set_xlabel('Actual')
            axes[1, 1].set_ylabel('Predicted')
            axes[1, 1].set_title('Actual vs Predicted', fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Marketing Mix Modeling Platform | Built with Streamlit</p>
    <p>Featuring scipy.optimize budget optimization & promotion variable support</p>
</div>
""", unsafe_allow_html=True)
