import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import statsmodels.api as sm
from scipy import stats
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

def clean_numeric_columns(df):
    """Clean numeric columns by removing commas and converting to float"""
    df_cleaned = df.copy()
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == 'object':
            try:
                # Try to convert after removing commas
                df_cleaned[col] = df_cleaned[col].astype(str).str.replace(',', '').astype(float)
            except (ValueError, AttributeError):
                # If conversion fails, keep as is
                pass
    return df_cleaned

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

def prepare_data_for_modeling(df, date_col, media_cols, target_col):
    """Prepare data with weekly aggregation"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['week'] = df[date_col].dt.to_period('W').dt.to_timestamp()
    
    # Aggregate to weekly level
    agg_dict = {target_col: 'sum'}
    for col in media_cols:
        agg_dict[col] = 'sum'
    
    weekly_data = df.groupby('week').agg(agg_dict).reset_index()
    weekly_data.columns = ['date'] + [target_col] + media_cols
    
    return weekly_data

def add_seasonality_features(df, date_col):
    """Add seasonality features: day of week and month (for daily data)"""
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
    - Upload media spend data
    - Analyze marketing effectiveness
    - Optimize budget allocation
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
                    
                    # Fill NaN with 0 for cost columns
                    cost_cols = [col for col in combined.columns if 'cost' in col.lower() or 'spend' in col.lower()]
                    combined[cost_cols] = combined[cost_cols].fillna(0)
                    
                    # Clean numeric columns (remove commas, convert to float)
                    combined = clean_numeric_columns(combined)
                    
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
        
        validation_col1, validation_col2 = st.columns(2)
        
        with validation_col1:
            # Check for minimum 20 months
            if date_range_months >= 20:
                st.success(f"✅ Sufficient data: {date_range_months:.1f} months (≥20 months required)")
            else:
                st.error(f"❌ Insufficient data: {date_range_months:.1f} months (<20 months)")
        
        with validation_col2:
            # Check for required columns
            has_revenue = any('revenue' in col.lower() for col in df.columns)
            if has_revenue:
                st.success("✅ Revenue column found")
            else:
                st.error("❌ Revenue column not found")
        
        # Display combined data
        st.markdown("---")
        st.markdown("### 📊 Combined Dataset")
        
        # Styled dataframe - only apply gradient to numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != date_col]
        
        if numeric_cols:
            st.dataframe(
                df.style.background_gradient(subset=numeric_cols, cmap='Blues'),
                use_container_width=True,
                height=400
            )
        else:
            st.dataframe(df, use_container_width=True, height=400)
        
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
        
        # Get numeric columns excluding date
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(14, 10))
            correlation_matrix = df[numeric_cols].corr()
            
            # Create heatmap with better visibility
            sns.heatmap(
                correlation_matrix, 
                annot=True, 
                fmt='.2f', 
                cmap='coolwarm', 
                center=0, 
                ax=ax,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8}
            )
            plt.title('Correlation Matrix: Media Channels & KPI', fontsize=16, fontweight='bold', pad=20)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show which columns are included
            with st.expander("ℹ️ Columns in Correlation Matrix"):
                st.write(f"**{len(numeric_cols)} numeric columns:**")
                st.write(", ".join(numeric_cols))
        else:
            st.warning(f"⚠️ Need at least 2 numeric columns for correlation. Found: {len(numeric_cols)}")
        
        # Spend vs Revenue plots for each media channel
        st.markdown("---")
        st.markdown("### 💰 Spend vs Revenue Analysis by Channel")
        st.info("These scatter plots show the relationship between media spend and revenue for each channel over the entire time period.")
        
        # Identify media spend columns and revenue column
        spend_cols = [col for col in df.columns if 'cost' in col.lower() or 'spend' in col.lower()]
        revenue_cols = [col for col in df.columns if 'revenue' in col.lower() or 'sales' in col.lower()]
        
        if spend_cols and revenue_cols:
            revenue_col = revenue_cols[0]  # Use first revenue column found
            
            # Create grid of plots
            num_channels = len(spend_cols)
            if num_channels > 0:
                cols_per_row = 2
                num_rows = (num_channels + cols_per_row - 1) // cols_per_row
                
                fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(14, 5*num_rows))
                if num_rows == 1 and cols_per_row == 1:
                    axes = np.array([[axes]])
                elif num_rows == 1:
                    axes = axes.reshape(1, -1)
                elif cols_per_row == 1:
                    axes = axes.reshape(-1, 1)
                
                for idx, spend_col in enumerate(spend_cols):
                    row = idx // cols_per_row
                    col = idx % cols_per_row
                    ax = axes[row, col]
                    
                    # Create scatter plot
                    x_data = df[spend_col].values
                    y_data = df[revenue_col].values
                    
                    # Remove any NaN or infinite values
                    mask = np.isfinite(x_data) & np.isfinite(y_data)
                    x_clean = x_data[mask]
                    y_clean = y_data[mask]
                    
                    if len(x_clean) > 0:
                        ax.scatter(x_clean, y_clean, alpha=0.6, s=50, color='steelblue')
                        
                        # Add trend line if enough data points
                        if len(x_clean) > 2:
                            z = np.polyfit(x_clean, y_clean, 1)
                            p = np.poly1d(z)
                            x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
                            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Trend')
                            ax.legend()
                        
                        # Calculate correlation
                        if len(x_clean) > 1:
                            corr = np.corrcoef(x_clean, y_clean)[0, 1]
                            ax.text(0.05, 0.95, f'Corr: {corr:.2f}', 
                                   transform=ax.transAxes, 
                                   fontsize=10, 
                                   verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    
                    channel_name = spend_col.replace('_Cost', '').replace('_cost', '').replace('_Spend', '').replace('_spend', '')
                    ax.set_title(f'{channel_name}', fontsize=12, fontweight='bold')
                    ax.set_xlabel('Spend', fontsize=10)
                    ax.set_ylabel('Revenue', fontsize=10)
                    ax.grid(True, alpha=0.3)
                
                # Hide empty subplots
                for idx in range(num_channels, num_rows * cols_per_row):
                    row = idx // cols_per_row
                    col = idx % cols_per_row
                    fig.delaxes(axes[row, col])
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("""
                **💡 What to look for:**
                - **Positive correlation**: Higher spend → Higher revenue (good!)
                - **Scattered points**: Inconsistent performance or other factors at play
                - **Flat trend**: Channel may not be driving incremental revenue
                - **Strong trend line**: Clear relationship between spend and outcomes
                """)
        else:
            st.warning("⚠️ Could not find spend and revenue columns for analysis.")

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
        
        if date_range_months < 20:
            st.error(f"❌ Insufficient data for modeling: {date_range_months:.1f} months available (20 months required)")
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
        
        # Model parameters
        st.markdown("---")
        st.markdown("### ⚙️ Model Parameters")
        
        # Option to use global or channel-specific parameters
        use_channel_specific = st.checkbox(
            "🎯 Use channel-specific parameters (Recommended for channels with different spend levels)",
            value=True,
            help="Allows different adstock and saturation curves per channel. Essential when channels have very different spend ranges."
        )
        
        if not use_channel_specific:
            # Global parameters (old approach)
            st.info("ℹ️ Using same parameters for all channels. Consider channel-specific for better accuracy.")
            param_col1, param_col2, param_col3 = st.columns(3)
            
            with param_col1:
                adstock_alpha = st.slider("Adstock Rate (α)", 0.0, 0.9, 0.5, 0.05, help="Carryover effect of advertising")
            
            with param_col2:
                hill_slope = st.slider("Hill Slope", 0.5, 2.0, 1.0, 0.1, help="Saturation curve shape")
            
            with param_col3:
                train_test_split = st.slider("Train/Test Split", 0.6, 0.9, 0.8, 0.05, help="Proportion of data for training")
            
            # Create dict with same params for all channels
            channel_params = {col: {'adstock': adstock_alpha, 'hill_slope': hill_slope} for col in media_cols}
            
        else:
            # Channel-specific parameters
            st.markdown("**📊 Set parameters for each channel:**")
            st.info("""
            💡 **Quick Guide:**
            - **High-spend channels** (Google): Higher adstock (0.6-0.8) + Lower slope (0.7-0.9) = Gentler saturation
            - **Low-spend channels** (Facebook): Lower adstock (0.3-0.5) + Higher slope (1.2-1.5) = Sharper saturation
            """)
            
            channel_params = {}
            
            # Create expandable sections for each channel
            for media_col in media_cols:
                channel_name = media_col.replace('_Cost', '').replace('_cost', '').replace('_Spend', '').replace('_spend', '')
                
                with st.expander(f"⚙️ {channel_name} Parameters", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        adstock = st.slider(
                            f"Adstock Rate",
                            0.0, 0.9, 0.5, 0.05,
                            key=f'adstock_{media_col}',
                            help=f"Carryover effect for {channel_name}. Higher = longer lasting impact."
                        )
                    
                    with col2:
                        slope = st.slider(
                            f"Hill Slope",
                            0.5, 2.0, 1.0, 0.1,
                            key=f'slope_{media_col}',
                            help=f"Saturation curve for {channel_name}. Lower = gentler, Higher = sharper."
                        )
                    
                    channel_params[media_col] = {'adstock': adstock, 'hill_slope': slope}
                    
                    # Show quick interpretation
                    if adstock > 0.6 and slope < 1.0:
                        st.success(f"✅ Configuration suggests: Long-lasting, gradual saturation (good for high-spend brand channel)")
                    elif adstock < 0.5 and slope > 1.2:
                        st.warning(f"⚠️ Configuration suggests: Quick impact, sharp saturation (good for performance channel)")
            
            # Train/test split (global)
            train_test_split = st.slider(
                "Train/Test Split", 
                0.6, 0.9, 0.8, 0.05, 
                help="Proportion of data for training",
                key='train_test_split_global'
            )
        
        # Add control variables option
        st.markdown("**Control Variables (Optional):**")
        control_cols = st.multiselect(
            "Select control variables",
            [col for col in df.columns if col not in media_cols and col != target_col and col != date_col],
            key='control_cols'
        )
        
        # Run model button
        st.markdown("---")
        if st.button("🚀 Run Marketing Mix Model", type="primary", use_container_width=True):
            with st.spinner("Training Marketing Mix Model... This may take a few minutes."):
                try:
                    # Clean data first (remove commas, convert to numeric)
                    st.info("Step 1/6: Cleaning and validating data...")
                    df = clean_numeric_columns(df)
                    
                    # Prepare daily data (no aggregation)
                    st.info("Step 2/6: Preparing daily data...")
                    daily_df = df.copy()
                    daily_df[date_col] = pd.to_datetime(daily_df[date_col])
                    daily_df = daily_df.sort_values(date_col).reset_index(drop=True)
                    
                    # Add seasonality (day of week + month)
                    st.info("Step 3/6: Adding seasonality features (day of week + month)...")
                    daily_df = add_seasonality_features(daily_df, date_col)
                    
                    # Engineer features
                    st.info("Step 4/6: Engineering media features (adstock + saturation)...")
                    
                    meta = {}
                    feat_cols = []
                    
                    for media_col in media_cols:
                        # Get channel-specific parameters
                        ch_adstock = channel_params[media_col]['adstock']
                        ch_slope = channel_params[media_col]['hill_slope']
                        
                        # Adstock with channel-specific rate
                        daily_df[f'{media_col}_adstock'] = adstock_transformation(
                            daily_df[media_col].values, alpha=ch_adstock
                        )
                        
                        # Hill saturation with channel-specific slope
                        kappa = np.nanmedian(daily_df[f'{media_col}_adstock'].values)
                        if not np.isfinite(kappa) or kappa <= 0:
                            kappa = np.nanmean(daily_df[f'{media_col}_adstock'].values) or 1.0
                        
                        daily_df[f'{media_col}_saturated'] = hill_transformation(
                            daily_df[f'{media_col}_adstock'].values,
                            kappa=kappa,
                            slope=ch_slope
                        )
                        
                        # Standardize
                        mu = daily_df[f'{media_col}_saturated'].mean()
                        sd = daily_df[f'{media_col}_saturated'].std() or 1.0
                        
                        feat_name = f'{media_col}_feat'
                        daily_df[feat_name] = (daily_df[f'{media_col}_saturated'] - mu) / sd
                        
                        feat_cols.append(feat_name)
                        
                        # Store metadata with channel-specific params
                        meta[feat_name] = {
                            'spend_col': media_col,
                            'kappa': kappa,
                            'slope': ch_slope,
                            'adstock': ch_adstock,
                            'mu': mu,
                            'sd': sd
                        }
                    
                    # Train/test split
                    st.info("Step 5/6: Splitting data into train and test sets...")
                    split_idx = int(len(daily_df) * train_test_split)
                    train_df = daily_df.iloc[:split_idx].copy()
                    test_df = daily_df.iloc[split_idx:].copy()
                    
                    # Prepare X and y
                    seasonality_cols = [col for col in daily_df.columns if 'dow_' in col or 'month_' in col]
                    
                    X_train = pd.concat([
                        pd.Series(1.0, index=train_df.index, name='const'),
                        train_df[feat_cols],
                        train_df[control_cols] if control_cols else pd.DataFrame(index=train_df.index),
                        train_df[seasonality_cols]
                    ], axis=1).astype('float64')
                    
                    X_test = pd.concat([
                        pd.Series(1.0, index=test_df.index, name='const'),
                        test_df[feat_cols],
                        test_df[control_cols] if control_cols else pd.DataFrame(index=test_df.index),
                        test_df[seasonality_cols]
                    ], axis=1).astype('float64')
                    
                    y_train = train_df[target_col].values.astype(float)
                    y_test = test_df[target_col].values.astype(float)
                    
                    # Train model
                    st.info("Step 6/6: Training OLS regression model...")
                    model = sm.OLS(y_train, X_train).fit()
                    
                    # Predictions
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    
                    # Calculate metrics
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
                    st.session_state.control_cols = control_cols
                    st.session_state.train_df = train_df
                    st.session_state.test_df = test_df
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.y_train_pred = y_train_pred
                    st.session_state.y_test_pred = y_test_pred
                    st.session_state.channel_params = channel_params  # Store all channel params
                    
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
                    ax1.plot(train_df['date'], y_train, label='Actual', color='green', alpha=0.7)
                    ax1.plot(train_df['date'], y_train_pred, label='Predicted', color='blue', alpha=0.7)
                    ax1.set_title(f'Training Set (R²={train_r2:.3f})', fontsize=14, fontweight='bold')
                    ax1.set_xlabel('Date')
                    ax1.set_ylabel(target_col)
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Test
                    ax2.plot(test_df['date'], y_test, label='Actual', color='green', alpha=0.7)
                    ax2.plot(test_df['date'], y_test_pred, label='Predicted', color='blue', alpha=0.7)
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
        control_cols = st.session_state.control_cols
        test_df = st.session_state.test_df
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        y_test_pred = st.session_state.y_test_pred
        channel_params = st.session_state.channel_params  # Get channel-specific params
        
        # Tabs for different analyses
        result_tabs = st.tabs([
            "📊 Channel Contribution",
            "💰 ROI Analysis",
            "📈 Response Curves",
            "🎯 Budget Allocation",
            "🔮 Monthly Forecast",
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
                
                # Marginal ROI at current spend (use channel-specific adstock)
                kappa = meta[feat]['kappa']
                slope = meta[feat]['slope']
                sd = meta[feat]['sd']
                ch_adstock = meta[feat]['adstock']  # Channel-specific adstock
                
                current_avg_spend = test_df[channel_name].mean()
                A = current_avg_spend / (1 - ch_adstock)
                
                marginal_roas = (beta / sd) * hill_derivative(A, kappa, slope) / (1 - ch_adstock)
                
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
            ch_adstock = meta[feat]['adstock']  # Channel-specific adstock
            
            # Generate spend range
            historical_spend = test_df[selected_channel].values
            max_spend = np.percentile(historical_spend, 95)
            spend_range = np.linspace(0, max_spend * 1.5, 200)
            
            # Calculate responses (use channel-specific adstock)
            adstocked = spend_range / (1 - ch_adstock)
            saturated = hill_transformation(adstocked, kappa, slope)
            standardized = (saturated - mu) / sd
            revenue = beta * standardized
            
            # Calculate marginal ROAS (use channel-specific adstock)
            marginal_roas = (beta / sd) * hill_derivative(adstocked, kappa, slope) / (1 - ch_adstock)
            
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
            axes[0, 0].set_xlabel('Weekly Spend', fontsize=11)
            axes[0, 0].set_ylabel('Incremental Revenue', fontsize=11)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Marginal ROAS
            axes[0, 1].plot(spend_range, marginal_roas, color='coral', linewidth=2)
            axes[0, 1].axvline(historical_spend.mean(), color='red', linestyle='--', label='Current avg spend')
            axes[0, 1].axhline(y=1, color='green', linestyle='--', label='Break-even')
            axes[0, 1].set_title('Marginal ROAS', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Weekly Spend', fontsize=11)
            axes[0, 1].set_ylabel('Marginal ROAS', fontsize=11)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # iROAS curve
            axes[1, 0].plot(spend_range[1:], iroas[1:], color='purple', linewidth=2)
            axes[1, 0].axvline(historical_spend.mean(), color='red', linestyle='--', label='Current avg spend')
            axes[1, 0].axhline(y=1, color='green', linestyle='--', label='Break-even')
            axes[1, 0].set_title('Incremental ROAS', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Weekly Spend', fontsize=11)
            axes[1, 0].set_ylabel('iROAS', fontsize=11)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Spend efficiency
            efficiency = revenue / spend_range
            efficiency[0] = 0
            axes[1, 1].plot(spend_range, efficiency, color='green', linewidth=2)
            axes[1, 1].axvline(historical_spend.mean(), color='red', linestyle='--', label='Current avg spend')
            axes[1, 1].set_title('Spend Efficiency', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Weekly Spend', fontsize=11)
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
        
        # Tab 4: Budget Allocation
        with result_tabs[3]:
            st.markdown("### Budget Allocation Optimizer")
            
            st.info("""
            This tool helps you optimize budget allocation across channels based on their marginal ROI.
            Channels with higher marginal ROI should receive more budget to maximize total revenue.
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
            
            # Simple allocation based on marginal ROI
            st.markdown("---")
            st.markdown("#### Recommended Allocation")
            
            # Calculate marginal ROI for each channel
            allocation_data = []
            
            for feat in feat_cols:
                channel_name = meta[feat]['spend_col']
                beta = float(model.params.get(feat, 0.0))
                kappa = meta[feat]['kappa']
                slope = meta[feat]['slope']
                sd = meta[feat]['sd']
                ch_adstock = meta[feat]['adstock']  # Channel-specific adstock
                
                current_spend = test_df[channel_name].mean()
                A = current_spend / (1 - ch_adstock)
                marginal_roas = (beta / sd) * hill_derivative(A, kappa, slope) / (1 - ch_adstock)
                
                allocation_data.append({
                    'Channel': channel_name.replace('_Cost', '').replace('_cost', ''),
                    'Current Weekly Spend': current_spend,
                    'Marginal ROI': marginal_roas,
                    'Current Total Spend': test_df[channel_name].sum()
                })
            
            alloc_df = pd.DataFrame(allocation_data)
            
            # Allocate proportionally to marginal ROI
            total_marginal_roi = alloc_df['Marginal ROI'].sum()
            alloc_df['Recommended %'] = (alloc_df['Marginal ROI'] / total_marginal_roi) * 100
            alloc_df['Recommended Budget'] = (alloc_df['Recommended %'] / 100) * new_budget
            alloc_df['Change vs Current'] = alloc_df['Recommended Budget'] - alloc_df['Current Total Spend']
            alloc_df['Change %'] = (alloc_df['Change vs Current'] / alloc_df['Current Total Spend']) * 100
            
            # Display
            st.dataframe(
                alloc_df.style.format({
                    'Current Weekly Spend': '{:,.0f}',
                    'Marginal ROI': '{:.2f}',
                    'Current Total Spend': '{:,.0f}',
                    'Recommended %': '{:.1f}%',
                    'Recommended Budget': '{:,.0f}',
                    'Change vs Current': '{:+,.0f}',
                    'Change %': '{:+.1f}%'
                }).background_gradient(subset=['Marginal ROI'], cmap='RdYlGn'),
                use_container_width=True
            )
            
            # Visualization
            st.markdown("---")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Current vs Recommended
            x = np.arange(len(alloc_df))
            width = 0.35
            
            ax1.bar(x - width/2, alloc_df['Current Total Spend'], width, label='Current', color='steelblue')
            ax1.bar(x + width/2, alloc_df['Recommended Budget'], width, label='Recommended', color='coral')
            ax1.set_xlabel('Channel')
            ax1.set_ylabel('Budget')
            ax1.set_title('Current vs Recommended Budget Allocation', fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(alloc_df['Channel'], rotation=45, ha='right')
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
            
            # Budget change
            colors = ['green' if x > 0 else 'red' for x in alloc_df['Change %']]
            ax2.barh(alloc_df['Channel'], alloc_df['Change %'], color=colors)
            ax2.set_xlabel('Change (%)')
            ax2.set_title('Recommended Budget Change by Channel', fontsize=14, fontweight='bold')
            ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            ax2.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Expected impact
            st.markdown("---")
            st.markdown("### 📊 Expected Impact")
            
            # Calculate expected revenue under new allocation
            # This is a simplified calculation
            expected_lift = sum([
                row['Marginal ROI'] * (row['Recommended Budget'] - row['Current Total Spend'])
                for _, row in alloc_df.iterrows()
            ])
            
            current_revenue = y_test.sum()
            expected_revenue = current_revenue + expected_lift
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Revenue", f"${current_revenue:,.0f}")
            
            with col2:
                st.metric("Expected Revenue", f"${expected_revenue:,.0f}", delta=f"${expected_lift:,.0f}")
            
            with col3:
                lift_pct = (expected_lift / current_revenue) * 100
                st.metric("Expected Lift", f"{lift_pct:+.1f}%")
        
        # Tab 5: Monthly Forecast
        with result_tabs[4]:
            st.markdown("### 🔮 Monthly Revenue Forecast")
            
            st.info("""
            This forecast projects monthly revenue based on:
            - Current media spend levels (average from test period)
            - Seasonal patterns (day of week + month effects from the model)
            - Baseline revenue
            """)
            
            # User inputs for forecast
            forecast_col1, forecast_col2 = st.columns(2)
            
            with forecast_col1:
                forecast_months = st.slider(
                    "Forecast Horizon (Months)",
                    min_value=1,
                    max_value=12,
                    value=3,
                    help="Number of months to forecast ahead"
                )
            
            with forecast_col2:
                spend_scenario = st.selectbox(
                    "Spend Scenario",
                    ["Current Levels", "Increase 10%", "Increase 20%", "Decrease 10%", "Decrease 20%", "Custom"],
                    help="Choose how to adjust spend for forecast"
                )
            
            # Custom spend adjustments if selected
            if spend_scenario == "Custom":
                st.markdown("**Custom Spend Adjustments per Channel:**")
                custom_adjustments = {}
                for col in media_cols:
                    channel_name = col.replace('_Cost', '').replace('_cost', '').replace('_Spend', '').replace('_spend', '')
                    adj = st.slider(
                        f"{channel_name} Adjustment",
                        -50, 100, 0, 5,
                        key=f'forecast_adj_{col}',
                        help=f"% change in {channel_name} spend"
                    )
                    custom_adjustments[col] = 1 + (adj / 100)
            
            if st.button("🚀 Generate Forecast", type="primary"):
                with st.spinner("Generating monthly forecast..."):
                    try:
                        # Get last date from test data
                        last_date = pd.to_datetime(test_df[date_col]).max()
                        
                        # Generate future dates (daily, then aggregate to monthly)
                        forecast_days = forecast_months * 30
                        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
                        
                        forecast_data = pd.DataFrame({'date': future_dates})
                        forecast_data['date'] = pd.to_datetime(forecast_data['date'])
                        
                        # Add seasonality features
                        forecast_data['day_of_week'] = forecast_data['date'].dt.dayofweek
                        forecast_data['month'] = forecast_data['date'].dt.month
                        
                        # Create day of week and month dummies
                        day_dummies = pd.get_dummies(forecast_data['day_of_week'], prefix='dow', drop_first=True)
                        month_dummies = pd.get_dummies(forecast_data['month'], prefix='month', drop_first=True)
                        forecast_data = pd.concat([forecast_data, day_dummies, month_dummies], axis=1)
                        
                        # Add media spend (based on scenario)
                        spend_multiplier = {
                            "Current Levels": 1.0,
                            "Increase 10%": 1.1,
                            "Increase 20%": 1.2,
                            "Decrease 10%": 0.9,
                            "Decrease 20%": 0.8
                        }
                        
                        for media_col in media_cols:
                            avg_spend = test_df[media_col].mean()
                            
                            if spend_scenario == "Custom":
                                forecast_data[media_col] = avg_spend * custom_adjustments[media_col]
                            else:
                                forecast_data[media_col] = avg_spend * spend_multiplier[spend_scenario]
                            
                            # Apply adstock and saturation transformations
                            ch_adstock = channel_params[media_col]['adstock']
                            ch_slope = channel_params[media_col]['hill_slope']
                            
                            feat = [f for f in feat_cols if meta[f]['spend_col'] == media_col][0]
                            kappa = meta[feat]['kappa']
                            mu = meta[feat]['mu']
                            sd = meta[feat]['sd']
                            
                            # Adstock
                            forecast_data[f'{media_col}_adstock'] = adstock_transformation(
                                forecast_data[media_col].values, alpha=ch_adstock
                            )
                            
                            # Saturation
                            forecast_data[f'{media_col}_saturated'] = hill_transformation(
                                forecast_data[f'{media_col}_adstock'].values,
                                kappa=kappa,
                                slope=ch_slope
                            )
                            
                            # Standardize
                            forecast_data[feat] = (forecast_data[f'{media_col}_saturated'] - mu) / sd
                        
                        # Add control variables (use mean from test)
                        if control_cols:
                            for ctrl in control_cols:
                                forecast_data[ctrl] = test_df[ctrl].mean()
                        
                        # Prepare X for prediction
                        seasonality_cols = [col for col in forecast_data.columns if 'dow_' in col or 'month_' in col]
                        
                        X_forecast = pd.concat([
                            pd.Series(1.0, index=forecast_data.index, name='const'),
                            forecast_data[feat_cols],
                            forecast_data[control_cols] if control_cols else pd.DataFrame(index=forecast_data.index),
                            forecast_data[seasonality_cols]
                        ], axis=1)
                        
                        # Ensure same columns as training
                        X_forecast = X_forecast.reindex(columns=X_train.columns, fill_value=0).astype('float64')
                        
                        # Predict
                        forecast_data['predicted_revenue'] = model.predict(X_forecast)
                        
                        # Aggregate to monthly
                        forecast_data['year_month'] = forecast_data['date'].dt.to_period('M')
                        monthly_forecast = forecast_data.groupby('year_month').agg({
                            'predicted_revenue': 'sum',
                            **{col: 'sum' for col in media_cols}
                        }).reset_index()
                        
                        monthly_forecast['year_month'] = monthly_forecast['year_month'].astype(str)
                        monthly_forecast.columns = ['Month', 'Forecasted Revenue'] + [f"{col.replace('_Cost', '').replace('_cost', '')} Spend" for col in media_cols]
                        
                        # Calculate total spend and ROI
                        spend_cols_renamed = [col for col in monthly_forecast.columns if 'Spend' in col]
                        monthly_forecast['Total Spend'] = monthly_forecast[spend_cols_renamed].sum(axis=1)
                        monthly_forecast['Forecast ROI'] = monthly_forecast['Forecasted Revenue'] / monthly_forecast['Total Spend']
                        
                        # Display results
                        st.markdown("---")
                        st.markdown("### 📊 Monthly Forecast Table")
                        
                        st.dataframe(
                            monthly_forecast.style.format({
                                'Forecasted Revenue': '{:,.0f}',
                                'Total Spend': '{:,.0f}',
                                'Forecast ROI': '{:.2f}',
                                **{col: '{:,.0f}' for col in spend_cols_renamed}
                            }).background_gradient(subset=['Forecasted Revenue', 'Forecast ROI'], cmap='Greens'),
                            use_container_width=True,
                            height=400
                        )
                        
                        # Summary metrics
                        st.markdown("---")
                        st.markdown("### 📈 Forecast Summary")
                        
                        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                        
                        with summary_col1:
                            total_forecast_revenue = monthly_forecast['Forecasted Revenue'].sum()
                            st.metric("Total Forecasted Revenue", f"${total_forecast_revenue:,.0f}")
                        
                        with summary_col2:
                            total_forecast_spend = monthly_forecast['Total Spend'].sum()
                            st.metric("Total Planned Spend", f"${total_forecast_spend:,.0f}")
                        
                        with summary_col3:
                            avg_monthly_revenue = monthly_forecast['Forecasted Revenue'].mean()
                            st.metric("Avg Monthly Revenue", f"${avg_monthly_revenue:,.0f}")
                        
                        with summary_col4:
                            overall_roi = total_forecast_revenue / total_forecast_spend if total_forecast_spend > 0 else 0
                            st.metric("Overall Forecast ROI", f"{overall_roi:.2f}")
                        
                        # Visualization
                        st.markdown("---")
                        st.markdown("### 📉 Forecast Visualization")
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        x_pos = range(len(monthly_forecast))
                        ax.bar(x_pos, monthly_forecast['Forecasted Revenue'], color='steelblue', alpha=0.7, label='Forecasted Revenue')
                        ax.set_xlabel('Month', fontsize=12)
                        ax.set_ylabel('Revenue', fontsize=12)
                        ax.set_title(f'Monthly Revenue Forecast - {spend_scenario} Scenario', fontsize=14, fontweight='bold')
                        ax.set_xticks(x_pos)
                        ax.set_xticklabels(monthly_forecast['Month'], rotation=45, ha='right')
                        ax.legend()
                        ax.grid(axis='y', alpha=0.3)
                        
                        # Add value labels on bars
                        for i, v in enumerate(monthly_forecast['Forecasted Revenue']):
                            ax.text(i, v, f'${v/1000:.0f}K', ha='center', va='bottom', fontsize=9)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Download button
                        st.markdown("---")
                        csv = monthly_forecast.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Forecast CSV",
                            data=csv,
                            file_name=f"mmm_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Error generating forecast: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        
        # Tab 6: Model Summary
        with result_tabs[5]:
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
                    'P-Value': model.pvalues[param]
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
            
            # Download model summary
            st.markdown("---")
            st.markdown("#### 📥 Export Results")
            
            # Prepare summary report
            summary_text = f"""
MARKETING MIX MODEL SUMMARY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== MODEL PERFORMANCE ===
R-squared: {model.rsquared:.4f}
Adjusted R-squared: {model.rsquared_adj:.4f}
MAPE (Test): {calculate_metrics(y_test, y_test_pred)[1]:.2%}
wMAPE (Test): {calculate_metrics(y_test, y_test_pred)[2]:.2%}

=== MODEL PARAMETERS (Channel-Specific) ===
Training Samples: {len(st.session_state.y_train)}
Test Samples: {len(y_test)}

Channel Parameters:
{chr(10).join([f"  {meta[f]['spend_col']}: Adstock={meta[f]['adstock']:.2f}, Slope={meta[f]['slope']:.2f}, Kappa={meta[f]['kappa']:.2f}" for f in feat_cols])}

=== CHANNEL CONTRIBUTIONS ===
{contrib_df.to_string()}

=== ROI ANALYSIS ===
{roi_df.to_string()}

"""
            
            st.download_button(
                label="📄 Download Summary Report",
                data=summary_text,
                file_name=f"mmm_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Marketing Mix Modeling Platform | Built with Streamlit</p>
    <p>For questions or support, contact your analytics team</p>
</div>
""", unsafe_allow_html=True)
