import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path

# Set wide page layout and custom theme
st.set_page_config(layout="wide", page_title="Crime Analysis Dashboard")

# Enhanced CSS for better visuals
st.markdown("""
    <style>
        .main { padding: 0rem 1rem; }
        .stPlotlyChart { margin-bottom: 2rem; }
        h1 { color: #2c3e50; text-align: center; margin-bottom: 2rem; }
        h2 { color: #34495e; margin-top: 1rem; }
        .metric-card {
            background-color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .grid-container { display: grid; gap: 1rem; }
    </style>
""", unsafe_allow_html=True)

def preprocess_data(df, file_type):
    """Preprocess dataframes based on their type"""
    if df is None:
        return None
    
    if file_type == 'violent_crimes':
        df['Year'] = df['Year'].astype(int)
        return df
    
    elif file_type == 'children_crimes':
        # Rename district column if it exists
        if 'DISTRICT' in df.columns:
            df = df.rename(columns={'DISTRICT': 'Area_Name'})
        elif 'STATE/UT' in df.columns:
            df = df.rename(columns={'STATE/UT': 'Area_Name'})  # Fixed missing parenthesis
        return df
    
    elif file_type == 'rape_victims':
        if 'Year' not in df.columns:
            return None
        df['Year'] = df['Year'].astype(int)
        return df
    
    elif file_type == 'police_complaints':
        if 'Area_Name' not in df.columns and 'STATE/UT' in df.columns:
            df = df.rename(columns={'STATE/UT': 'Area_Name'})
        return df
    
    return df

def create_violent_crimes_chart(df):
    if df is None or 'Year' not in df.columns:
        return None
    try:
        # Get the last numeric column for analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis_col = numeric_cols[-1]
            chart_data = df.groupby('Year')[analysis_col].sum().reset_index()
            fig = px.line(chart_data, x='Year', y=analysis_col,
                         title="Violent Crimes Trend Over Years")
            fig.update_layout(showlegend=False)
            return fig
    except Exception as e:
        st.error(f"Error creating violent crimes chart: {str(e)}")
    return None

def create_children_crimes_chart(df):
    if df is None:
        return None
    try:
        # Get the last numeric column for analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis_col = numeric_cols[-1]
            if 'Area_Name' in df.columns:
                chart_data = df.groupby('Area_Name')[analysis_col].sum().reset_index()
                chart_data = chart_data.nlargest(10, analysis_col)
                fig = px.bar(chart_data, x='Area_Name', y=analysis_col,
                           title="Top 10 Areas - Crimes Against Children")
                fig.update_layout(xaxis_tickangle=-45)
                return fig
    except Exception as e:
        st.error(f"Error creating children crimes chart: {str(e)}")
    return None

def load_crime_data():
    """Load and preprocess all crime-related CSV files"""
    crime_files = {
        'violent_crimes': '28_Trial_of_violent_crimes_by_courts.csv',
        'children_crimes': 'crime/03_District_wise_crimes_committed_against_children_2013.csv',
        'sc_crimes': 'crime/02_01_District_wise_crimes_committed_against_SC_2013.csv',
        'st_crimes': 'crime/02_District_wise_crimes_committed_against_ST_2013.csv',
        'ipc_crimes': 'crime/01_District_wise_crimes_committed_IPC_2013.csv'
    }
    
    data = {}
    # Try different possible paths for the data files
    possible_paths = [
        Path(__file__).parent,  # Local development path
        Path.cwd(),  # Current working directory
        Path('crime'),  # Direct crime folder
        Path('.'),  # Root directory
    ]

    for key, filepath in crime_files.items():
        df = None
        for base_path in possible_paths:
            try:
                full_path = base_path / filepath
                if full_path.exists():
                    df = pd.read_csv(full_path)
                    st.debug(f"Successfully loaded {filepath} from {full_path}")  # Changed from st.info to st.debug
                    break
                else:
                    # Try without the 'crime/' prefix if it exists
                    alt_path = base_path / filepath.replace('crime/', '')
                    if alt_path.exists():
                        df = pd.read_csv(alt_path)
                        st.debug(f"Successfully loaded {filepath} from {alt_path}")  # Changed from st.info to st.debug
                        break
            except Exception as e:
                continue
        
        if df is None:
            st.debug(f"Failed to load {filepath} from any location")  # Changed from st.error to st.debug
            data[key] = None
        else:
            data[key] = preprocess_data(df, key)
    
    return data

def create_state_crime_comparison(children_df, sc_df, st_df):
    """Create state-wise crime comparison visualization"""
    # Function to get total crimes from a dataframe
    def get_total_crimes(df):
        if 'Total' in df.columns:
            return df.groupby('STATE/UT')['Total'].sum()
        # If 'Total' doesn't exist, sum all numeric columns except 'Year'
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'Year']
        return df.groupby('STATE/UT')[numeric_cols].sum().sum(axis=1)

    # Create the comparison dataframe
    states = pd.concat([
        get_total_crimes(children_df).rename('Children Crimes'),
        get_total_crimes(sc_df).rename('SC Crimes'),
        get_total_crimes(st_df).rename('ST Crimes')
    ], axis=1).fillna(0)
    
    fig = go.Figure()
    for col in states.columns:
        fig.add_trace(go.Bar(
            name=col,
            x=states.index,
            y=states[col],
            text=states[col].round(0),
            textposition='auto',
        ))
    
    fig.update_layout(
        title="Crime Comparison Across States",
        barmode='group',
        height=600,
        xaxis_tickangle=-45
    )
    return fig

def create_crime_trend_analysis(violent_crimes_df):
    """Analyze violent crime trends over years"""
    if violent_crimes_df is None:
        return None
    
    try:
        # Get numeric columns for analysis
        numeric_cols = violent_crimes_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            yearly_trends = violent_crimes_df.groupby(['Year', 'Group_Name'])[numeric_cols[-2:]].sum().reset_index()
            
            fig = px.line(yearly_trends,
                         x='Year',
                         y=numeric_cols[-2:],
                         color='Group_Name',
                         title="Violent Crime Trends Over Years")
            return fig
    except Exception as e:
        st.error(f"Error creating trend analysis: {str(e)}")
    return None

def create_crime_type_distribution(ipc_df):
    """Create crime type distribution analysis"""
    crime_cols = [col for col in ipc_df.columns if col not in ['STATE/UT', 'DISTRICT', 'Year', 'Total']]
    crime_totals = ipc_df[crime_cols].sum().sort_values(ascending=False)
    
    fig = px.pie(values=crime_totals.values,
                 names=crime_totals.index,
                 title="Distribution of Crime Types",
                 hole=0.4)
    return fig

def create_dashboard():
    st.title("🏛️ Comprehensive Indian Crime Analysis Dashboard")
    
    data = load_crime_data()
    
    # Top metrics row
    metrics = st.columns(4)
    with metrics[0]:
        if data['violent_crimes'] is not None and not data['violent_crimes'].empty:
            total_violent = data['violent_crimes'][data['violent_crimes'].select_dtypes(include=[np.number]).columns[-1]].sum()
            st.metric("Total Violent Crimes", f"{int(total_violent):,}")
        else:
            st.metric("Total Violent Crimes", "N/A")
    
    with metrics[1]:
        if data['children_crimes'] is not None and 'Total' in data['children_crimes'].columns:
            total_children = data['children_crimes']['Total'].sum()
            st.metric("Crimes Against Children", f"{int(total_children):,}")
        else:
            st.metric("Crimes Against Children", "N/A")
    
    with metrics[2]:
        if data['sc_crimes'] is not None:
            # Get all numeric columns except 'Year' for SC crimes
            numeric_cols = data['sc_crimes'].select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != 'Year']
            if len(numeric_cols) > 0:
                if 'Total' in numeric_cols:
                    total_sc = data['sc_crimes']['Total'].sum()
                else:
                    total_sc = data['sc_crimes'][numeric_cols].sum().sum()
                st.metric("Crimes Against SC", f"{int(total_sc):,}")
            else:
                st.metric("Crimes Against SC", "N/A")
        else:
            st.metric("Crimes Against SC", "N/A")
    
    with metrics[3]:
        if data['st_crimes'] is not None:
            # Get all numeric columns except 'Year' for ST crimes
            numeric_cols = data['st_crimes'].select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != 'Year']
            if len(numeric_cols) > 0:
                if 'Total' in numeric_cols:
                    total_st = data['st_crimes']['Total'].sum()
                else:
                    total_st = data['st_crimes'][numeric_cols].sum().sum()
                st.metric("Crimes Against ST", f"{int(total_st):,}")
            else:
                st.metric("Crimes Against ST", "N/A")
        else:
            st.metric("Crimes Against ST", "N/A")

    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["State Analysis", "Crime Trends", "Detailed Analysis"])
    
    with tab1:
        st.subheader("State-Level Crime Analysis")
        
        # Define crime_cols at the start of tab1
        crime_cols = [col for col in data['ipc_crimes'].columns 
                     if col not in ['STATE/UT', 'DISTRICT', 'Year', 'Total']]
        
        # First row - Existing visualizations
        col1, col2 = st.columns([2, 1])
        with col1:
            state_comparison = create_state_crime_comparison(
                data['children_crimes'], 
                data['sc_crimes'], 
                data['st_crimes']
            )
            st.plotly_chart(state_comparison, use_container_width=True)
        
        with col2:
            crime_distribution = create_crime_type_distribution(data['ipc_crimes'])
            st.plotly_chart(crime_distribution, use_container_width=True)
        
        # Second row - New state-level visualizations
        row2_cols = st.columns(2)
        with row2_cols[0]:
            # Calculate state totals from all numeric columns except Year
            numeric_cols = [col for col in data['ipc_crimes'].columns 
                          if col not in ['STATE/UT', 'DISTRICT', 'Year']
                          and np.issubdtype(data['ipc_crimes'][col].dtype, np.number)]
            state_totals = data['ipc_crimes'].groupby('STATE/UT')[numeric_cols].sum().sum(axis=1)
            
            # Create state-wise map for India
            fig = go.Figure(data=go.Choropleth(
                geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
                featureidkey='properties.ST_NM',
                locations=state_totals.index,
                z=state_totals.values,
                colorscale='Reds',
                colorbar_title="Total Crimes"
            ))

            fig.update_geos(
                visible=False,
                projection=dict(
                    type='mercator',
                    scale=1
                ),
                center=dict(lat=20.5937, lon=78.9629),
                fitbounds="locations"
            )
            
            fig.update_layout(
                title="Crime Density Across States",
                geo_scope='asia',
                width=800,
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with row2_cols[1]:
            # Top 10 states comparison
            top_states = state_totals.nlargest(10)
            fig = px.bar(
                x=top_states.index, 
                y=top_states.values,
                title="Top 10 States by Total Crime Rate",
                labels={'x': 'State', 'y': 'Total Crimes'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        # Third row - Additional state insights
        row3_cols = st.columns(2)
        with row3_cols[0]:
            # Improved crime type distribution by state
            crime_by_state = data['ipc_crimes'].groupby('STATE/UT')[crime_cols].sum()
            # Normalize the data for better visualization
            crime_by_state_normalized = (crime_by_state - crime_by_state.min()) / (crime_by_state.max() - crime_by_state.min())
            
            fig = px.imshow(
                crime_by_state_normalized,
                title="Crime Type Distribution by State (Normalized)",
                labels=dict(x="Crime Type", y="State", color="Normalized Count"),
                aspect="auto",
                color_continuous_scale="RdYlBu_r"
            )
            fig.update_layout(
                height=800,  # Increase height for better readability
                xaxis_tickangle=-45,
                yaxis={'dtick': 1},  # Show all state labels
            )
            fig.update_traces(hoverongaps=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add a note about normalization
            st.info("Note: Values are normalized for better visualization of patterns across different crime types")
        
        with row3_cols[1]:
            # State clustering with error handling for sklearn
            try:
                from sklearn.preprocessing import StandardScaler
                scaled_data = StandardScaler().fit_transform(crime_by_state)
                fig = px.scatter(
                    x=scaled_data[:, 0],
                    y=scaled_data[:, 1],
                    text=crime_by_state.index,
                    title="State Clustering by Crime Patterns"
                )
            except ImportError:
                # Alternative visualization when sklearn is not available
                # Create a simple scatter plot using raw data
                crime_sums = crime_by_state.sum(axis=1)
                crime_means = crime_by_state.mean(axis=1)
                fig = px.scatter(
                    x=crime_sums,
                    y=crime_means,
                    text=crime_by_state.index,
                    title="State Crime Patterns (Total vs Average)",
                    labels={
                        'x': 'Total Crimes',
                        'y': 'Average Crimes per Category'
                    }
                )
                st.info("📌 Note: Install scikit-learn for advanced clustering analysis: `pip install scikit-learn`")
            
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Temporal Crime Analysis")
        
        # First row - Main trends
        col1, col2 = st.columns(2)
        with col1:
            trend_analysis = create_crime_trend_analysis(data['violent_crimes'])
            st.plotly_chart(trend_analysis, use_container_width=True)
        
        with col2:
            # Year-over-year comparison (existing code)
            if data['violent_crimes'] is not None and 'Year' in data['violent_crimes'].columns:
                # Get the last numeric column for analysis
                numeric_cols = data['violent_crimes'].select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    analysis_col = numeric_cols[-1]
                    yoy_data = data['violent_crimes'].groupby('Year')[analysis_col].sum()
                    yoy_change = yoy_data.pct_change() * 100
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=yoy_change.index,
                        y=yoy_change.values,
                        name='YoY Change %'
                    ))
                    fig.update_layout(
                        title="Year-over-Year Change in Crime Rate",
                        yaxis_title="Percent Change (%)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No numeric data available for year-over-year comparison")
            else:
                st.write("Year-wise data not available for comparison")

        # New row - Additional temporal analysis
        st.subheader("Advanced Temporal Analysis")
        temporal_cols = st.columns(2)
        
        with temporal_cols[0]:
            # Cumulative crime trends
            if data['violent_crimes'] is not None and 'Year' in data['violent_crimes'].columns:
                numeric_cols = data['violent_crimes'].select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    analysis_col = numeric_cols[-1]
                    cumulative_data = data['violent_crimes'].groupby('Year')[analysis_col].sum().cumsum()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=cumulative_data.index,
                        y=cumulative_data.values,
                        mode='lines+markers',
                        name='Cumulative Total',
                        fill='tonexty'
                    ))
                    fig.update_layout(
                        title="Cumulative Crime Progression",
                        xaxis_title="Year",
                        yaxis_title="Cumulative Total Cases"
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with temporal_cols[1]:
            # Monthly/Seasonal distribution if available
            if 'Month' in data['violent_crimes'].columns:
                monthly_data = data['violent_crimes'].groupby('Month')[analysis_col].mean()
                fig = px.line_polar(r=monthly_data.values, 
                                  theta=monthly_data.index,
                                  line_close=True,
                                  title="Monthly Crime Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Yearly distribution visualization
                yearly_dist = data['violent_crimes'].groupby('Year')[analysis_col].sum()
                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=yearly_dist.values,
                    name='Yearly Distribution'
                ))
                fig.update_layout(title="Yearly Crime Distribution")
                st.plotly_chart(fig, use_container_width=True)

        # Third row - Comparative analysis
        st.subheader("Comparative Trend Analysis")
        trend_cols = st.columns(2)
        
        with trend_cols[0]:
            # Growth rate analysis
            if data['violent_crimes'] is not None:
                yearly_data = data['violent_crimes'].groupby('Year')[analysis_col].sum()
                growth_rate = yearly_data.pct_change() * 100
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=growth_rate.index,
                    y=growth_rate.values,
                    mode='lines+markers',
                    name='Growth Rate'
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                fig.update_layout(
                    title="Year-over-Year Growth Rate",
                    yaxis_title="Growth Rate (%)",
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)

        with trend_cols[1]:
            # Moving averages comparison
            if data['violent_crimes'] is not None:
                yearly_data = data['violent_crimes'].groupby('Year')[analysis_col].sum()
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=yearly_data.index,
                    y=yearly_data.values,
                    mode='lines',
                    name='Actual'
                ))
                fig.add_trace(go.Scatter(
                    x=yearly_data.index,
                    y=yearly_data.rolling(window=2).mean(),
                    mode='lines',
                    name='2-Year MA'
                ))
                fig.add_trace(go.Scatter(
                    x=yearly_data.index,
                    y=yearly_data.rolling(window=3).mean(),
                    mode='lines',
                    name='3-Year MA'
                ))
                fig.update_layout(
                    title="Moving Average Comparison",
                    xaxis_title="Year",
                    yaxis_title="Number of Cases"
                )
                st.plotly_chart(fig, use_container_width=True)

        # Add trend insights
        st.subheader("📈 Trend Insights")
        insight_cols = st.columns(3)
        with insight_cols[0]:
            if data['violent_crimes'] is not None:
                latest_year = yearly_data.index.max()
                latest_value = yearly_data.iloc[-1]
                prev_value = yearly_data.iloc[-2]
                change = ((latest_value - prev_value) / prev_value) * 100
                st.metric("Latest Year Trend", 
                         f"{latest_value:,.0f}",
                         f"{change:+.1f}%")

        with insight_cols[1]:
            if data['violent_crimes'] is not None:
                avg_last_3 = yearly_data.tail(3).mean()
                avg_prev_3 = yearly_data.iloc[-6:-3].mean()
                period_change = ((avg_last_3 - avg_prev_3) / avg_prev_3) * 100
                st.metric("3-Year Average", 
                         f"{avg_last_3:,.0f}",
                         f"{period_change:+.1f}%")

        with insight_cols[2]:
            if data['violent_crimes'] is not None:
                trend = "🔺 Increasing" if period_change > 0 else "🔻 Decreasing"
                confidence = abs(period_change)
                st.metric("Overall Trend", 
                         trend,
                         f"Confidence: {min(100, confidence):,.1f}%")

    with tab3:
        # Add interactive filters
        st.subheader("Interactive Analysis")
        col_filters = st.columns(3)
        with col_filters[0]:
            selected_state = st.selectbox("Select State", data['ipc_crimes']['STATE/UT'].unique())
        with col_filters[1]:
            crime_type_options = [col for col in data['ipc_crimes'].columns 
                                if col not in ['STATE/UT', 'DISTRICT', 'Year', 'Total']]
            selected_crime = st.selectbox("Select Crime Type", crime_type_options)
        with col_filters[2]:
            chart_type = st.selectbox("Select Chart Type", 
                                    ["Bar Chart", "Heat Map", "Bubble Chart", "Line Chart", 
                                     "Sunburst", "Box Plot", "Violin Plot", "3D Scatter"])
        
        # Initialize state_data first
        state_data = data['ipc_crimes'][data['ipc_crimes']['STATE/UT'] == selected_state]
        state_data = state_data[state_data['DISTRICT'] != 'ZZ TOTAL']
        crime_cols = [col for col in state_data.columns 
                     if col not in ['STATE/UT', 'DISTRICT', 'Year', 'Total']]
            
        # Add time range filter if applicable
        if 'Year' in state_data.columns:
            year_range = st.slider("Select Year Range", 
                                 min_value=int(state_data['Year'].min()),
                                 max_value=int(state_data['Year'].max()),
                                 value=(int(state_data['Year'].min()), int(state_data['Year'].max())))
            state_data = state_data[(state_data['Year'] >= year_range[0]) & 
                                  (state_data['Year'] <= year_range[1])]

        # First row of visualizations
        row1_cols = st.columns(2)
        with row1_cols[0]:
            if chart_type == "Bar Chart":
                fig = px.bar(
                    state_data.melt(id_vars=['DISTRICT'], value_vars=crime_cols),
                    x='DISTRICT', y='value', color='variable',
                    title=f"Crime Distribution in {selected_state}",
                    barmode='stack'
                )
            elif chart_type == "Heat Map":
                heat_data = state_data.pivot_table(
                    values=selected_crime, 
                    index='DISTRICT',
                    aggfunc='sum'
                )
                fig = px.imshow(heat_data.T,
                    title=f"Heat Map of {selected_crime} in {selected_state}")
            elif chart_type == "Bubble Chart":
                fig = px.scatter(state_data,
                    x='DISTRICT', y=selected_crime,
                    size=selected_crime,
                    title=f"Bubble Chart of {selected_crime} in {selected_state}")
            elif chart_type == "Sunburst":
                fig = px.sunburst(state_data,
                    path=['DISTRICT'], values=selected_crime,
                    title=f"Sunburst of {selected_crime} in {selected_state}")
            elif chart_type == "Box Plot":
                fig = px.box(state_data,
                    x='DISTRICT', y=selected_crime,
                    title=f"Box Plot of {selected_crime} by District")
            elif chart_type == "Violin Plot":
                fig = px.violin(state_data,
                    x='DISTRICT', y=selected_crime,
                    title=f"Violin Plot of {selected_crime} by District")
            elif chart_type == "3D Scatter":
                # Get top 3 crime types for 3D visualization
                top_crimes = crime_cols[:3]
                fig = px.scatter_3d(state_data,
                    x=top_crimes[0], y=top_crimes[1], z=top_crimes[2],
                    color='DISTRICT',
                    title=f"3D Crime Analysis in {selected_state}")
            else:  # Line Chart
                fig = px.line(state_data,
                    x='DISTRICT', y=selected_crime,
                    title=f"Trend of {selected_crime} in {selected_state}")
            
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        # Add correlation analysis
        if st.checkbox("Show Correlation Analysis"):
            correlation = state_data[crime_cols].corr()
            fig = px.imshow(correlation,
                title="Crime Type Correlation Matrix",
                labels=dict(color="Correlation"))
            st.plotly_chart(fig, use_container_width=True)

        # Add crime trends over time if Year column exists
        if 'Year' in state_data.columns and st.checkbox("Show Time Series Analysis"):
            fig = px.line(state_data.groupby('Year')[selected_crime].mean(),
                title=f"Time Series Analysis of {selected_crime}")
            fig.add_scatter(x=state_data.groupby('Year')[selected_crime].mean().index,
                          y=state_data.groupby('Year')[selected_crime].mean().rolling(3).mean(),
                          name='Moving Average')
            st.plotly_chart(fig, use_container_width=True)

        # Enhanced district comparison
        numeric_cols = [col for col in state_data.columns 
                      if col not in ['STATE/UT', 'DISTRICT', 'Year']
                      and np.issubdtype(state_data[col].dtype, np.number)]
        
        if 'Total' in state_data.columns:
            district_totals = state_data.groupby('DISTRICT')['Total'].sum()
        else:
            district_totals = state_data.groupby('DISTRICT')[numeric_cols].sum().sum(axis=1)
        
        district_totals = district_totals.sort_values(ascending=True)
        
        fig = px.bar(
            x=district_totals.values,
            y=district_totals.index,
            orientation='h',
            title=f"Total Crimes by District in {selected_state}"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Second row of visualizations
        row2_cols = st.columns(2)
        with row2_cols[0]:
            # Time series analysis if Year column exists
            if 'Year' in state_data.columns:
                # Filter out ZZ TOTAL for yearly trend
                yearly_trend = state_data[state_data['DISTRICT'] != 'ZZ TOTAL'].groupby('Year')[selected_crime].sum()
                fig = px.line(yearly_trend, 
                    title=f"Yearly Trend of {selected_crime} in {selected_state}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Crime type comparison
                crime_comparison = state_data[crime_cols].sum()
                fig = px.pie(values=crime_comparison.values,
                            names=crime_comparison.index,
                            title="Crime Type Distribution")
                st.plotly_chart(fig, use_container_width=True)

        with row2_cols[1]:
            # District-wise crime rate per category (excluding ZZ TOTAL)
            crime_rate = state_data[state_data['DISTRICT'] != 'ZZ TOTAL'].groupby('DISTRICT')[selected_crime].sum()
            fig = px.bar(crime_rate,
                title=f"{selected_crime} Rate by District",
                labels={'value': 'Crime Rate', 'DISTRICT': 'District'})
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        # Add statistical insights
        st.subheader("Statistical Insights")
        stats_cols = st.columns(3)
        with stats_cols[0]:
            st.metric("Highest Crime District", 
                     district_totals.index[-1],
                     f"{district_totals.iloc[-1]:,.0f} cases")
        with stats_cols[1]:
            st.metric("Average Crime Rate", 
                     f"{district_totals.mean():,.0f}",
                     f"{district_totals.std():,.0f} std dev")
        with stats_cols[2]:
            st.metric(f"Total {selected_crime}", 
                     f"{state_data[selected_crime].sum():,.0f}",
                     f"{state_data[selected_crime].mean():,.0f} avg")

    # Footer with insights
    st.markdown("---")
    st.subheader("📊 Key Insights")
    
    insights_cols = st.columns(3)
    with insights_cols[0]:
        st.markdown("### Geographical Patterns")
        st.write("• Higher crime rates in metropolitan areas")
        st.write("• Significant variation across states")
        st.write("• Urban-rural crime pattern differences")

    with insights_cols[1]:
        st.markdown("### Temporal Trends")
        st.write("• Year-over-year changes in crime rates")
        st.write("• Seasonal variations in specific crimes")
        st.write("• Long-term trend analysis")

    with insights_cols[2]:
        st.markdown("### Vulnerable Groups")
        st.write("• Analysis of crimes against children")
        st.write("• SC/ST targeted crime patterns")
        st.write("• Gender-based crime distribution")

if __name__ == "__main__":
    create_dashboard()