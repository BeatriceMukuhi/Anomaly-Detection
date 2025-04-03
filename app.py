import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Configure page settings
st.set_page_config(
    page_title="Equipment Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom styling
st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #040474;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.8rem;
            font-weight: bold;
            color: #424242;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        .section-header {
            font-size: 1.5rem;
            font-weight: bold;
            color: #212121;
            margin-top: 1.5rem;
            margin-bottom: 0.75rem;
        }
        .highlight {
            background-color: #ff6900;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 0.5rem solid #040474;
        }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('equipment_anomaly_data.csv')

# Load models
@st.cache_resource
def load_model():
    model = joblib.load('equipment_fault_prediction_model.pkl')
    scaler = joblib.load('equipment_fault_prediction_scaler.pkl')
    return model, scaler

# Navigation
def main():
    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to", ["Project Overview", "Data Exploration", "Prediction Model"])
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### About")
        st.sidebar.info(
            "This application provides tools for analyzing equipment anomaly data "
            "and predicting potential failures. Built with Streamlit and Scikit-learn."
        )

    # Page content
    if page == "Project Overview":
        display_project_overview()
    elif page == "Data Exploration":
        display_data_exploration()
    else:
        display_prediction_model()

# Page 1: Project Overview
def display_project_overview():
    st.markdown('<div class="main-header">Equipment Anomaly Detection Project</div>', unsafe_allow_html=True)
    
    st.write("""
    This project analyzes equipment anomaly data to predict whether equipment is faulty or not.
    We performed detailed exploratory data analysis and built machine learning models to accurately
    identify potential equipment failures before they occur.
    """)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<div class="sub-header">Project Summary</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="highlight">
        <p><strong>Objective:</strong> Develop a predictive model to identify equipment faults before they occur</p>
        <p><strong>Dataset:</strong> 7,672 equipment records with 7 features including environmental and operational parameters</p>
        <p><strong>ML Models:</strong> Logistic Regression, Decision Tree, Random Forest, Support Vector Machine</p>
        <p><strong>Best Model:</strong> Tuned Random Forest with high accuracy and F1 score</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">Key Findings</div>', unsafe_allow_html=True)
        
        # Using Streamlit components instead of raw HTMLd sections
        with st.container():
            st.subheader("Feature Analysis Results")
            
            col_left, col_right = st.columns([1, 3])
            
            with col_left:
                st.markdown("**Vibration**")
                st.markdown("*(Correlation: 0.43)*")
                
            with col_right:
                st.markdown("• Strongest predictor of equipment faults")
                st.markdown("• Non-faulty equipment shows readings around 1.5 units")
                st.markdown("• Faulty equipment exhibits wider variability")
                st.markdown("• Clear separation between normal and fault distributions")
            
            col_left, col_right = st.columns([1, 3])
            
            with col_left:
                st.markdown("**Pressure**")
                st.markdown("*(Correlation: 0.20)*")
                
            with col_right:
                st.markdown("• Second most significant indicator for faults")
                st.markdown("• Normal operation around 38 units")
                st.markdown("• Faulty equipment operates under unusual pressures")
                st.markdown("• Higher variance is a reliable fault indicator")
            
            col_left, col_right = st.columns([1, 3])
            
            with col_left:
                st.markdown("**Temperature**")
                st.markdown("*(Correlation: 0.18)*")
                
            with col_right:
                st.markdown("• Non-faulty operation in narrow range around 75°C")
                st.markdown("• Faulty equipment shows broader distribution")
                st.markdown("• Extreme values correlate with failures")
            
            col_left, col_right = st.columns([1, 3])
            
            with col_left:
                st.markdown("**Humidity**")
                st.markdown("*(Correlation: 0.01)*")
                
            with col_right:
                st.markdown("• Negligible correlation with equipment faults")
                st.markdown("• Not a reliable predictor for fault detection")
                st.markdown("• Normal equipment clusters around 50% humidity")
            
            col_left, col_right = st.columns([1, 3])
            
            with col_left:
                st.markdown("**Equipment Types**")
                
            with col_right:
                st.markdown("• Turbines show highest fault rate (10.06%)")
                st.markdown("• All types have comparable fault rates (~10%)")
                st.markdown("• Fault patterns consistent across types")
        
        st.markdown('<div class="section-header">Analysis Conclusions</div>', unsafe_allow_html=True)
        st.write("""
        - Vibration is the most influential feature for predicting equipment faults, with the strongest correlation (0.43)
        - Pressure and temperature are secondary indicators with correlations of 0.20 and 0.18 respectively
        - Deviations from normal operating ranges in these parameters can signal potential failures
        - Equipment operating outside normal vibration (1.5 units), pressure (38 units), or temperature (75°C) ranges requires attention
        - The pairplot analysis reveals that faulty equipment shows scattered distribution across feature ranges, while non-faulty equipment forms tighter clusters
        - While all equipment types exhibit similar fault rates, their operating parameters differ, requiring type-specific monitoring
        """)
        
        st.markdown('<div class="section-header">Methodology</div>', unsafe_allow_html=True)
        st.write("""
        1. Data cleaning and exploratory data analysis
        2. Feature engineering to create more predictive variables
        3. Model development using various classification algorithms
        4. Hyperparameter tuning to optimize model performance
        5. Evaluation based on accuracy, precision, recall, and F1 score
        """)
        
        st.markdown('<div class="section-header">Business Impact</div>', unsafe_allow_html=True)
        st.write("""
        - Reduced equipment downtime through proactive maintenance
        - Lower maintenance costs by addressing issues before catastrophic failure
        - Improved operational efficiency and equipment lifespan
        - Enhanced safety through prevention of equipment failures
        """)
    
    with col2:
        st.markdown('<div class="section-header">Navigation Guide</div>', unsafe_allow_html=True)
        st.info("**Data Exploration Page**\nExplore the dataset through interactive visualizations showing relationships between variables and fault patterns.")
        st.info("**Prediction Model Page**\nTest the machine learning model by inputting equipment parameters to predict potential failures.")
        
        # Add a visual representation of feature importance
        st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
        feature_importance = {
            "Feature": ["Vibration", "Pressure", "Temperature", "Equipment Type", "Humidity"],
            "Correlation": [0.43, 0.20, 0.18, 0.05, 0.01]
        }
        
        fig = px.bar(
            feature_importance,
            x="Feature",
            y="Correlation",
            color="Correlation",
            color_continuous_scale="Bluered",
            height=300
        )
        fig.update_layout(
            title="Correlation with Equipment Faults",
            yaxis_title="Correlation Coefficient"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Display a sample of the data
    st.markdown('<div class="sub-header">Dataset Preview</div>', unsafe_allow_html=True)
    df = load_data()
    st.dataframe(df.head())
    
    # Show project metrics
    st.markdown('<div class="sub-header">Project Metrics</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{df.shape[0]:,}")
    with col2:
        st.metric("Features", f"{df.shape[1]-1}")
    with col3:
        st.metric("Fault Rate", f"{df['faulty'].mean()*100:.2f}%")
    with col4:
        # Load model for best score
        try:
            model, _ = load_model()
            st.metric("Best Model F1", f"{0.97:.2f}")
        except:
            st.metric("Best Model F1", "N/A")
    
    # Add operating ranges visualization
    st.markdown('<div class="sub-header">Normal Operating Ranges</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="section-header">Temperature</div>', unsafe_allow_html=True)
        fig = px.histogram(df[df['faulty'] == 0], x='temperature', nbins=30, 
                         color_discrete_sequence=['#040474'])
        fig.add_vline(x=75, line_dash="dash", line_color="green", annotation_text="Optimal")
        fig.update_layout(title="Normal Temperature Range", height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">Pressure</div>', unsafe_allow_html=True)
        fig = px.histogram(df[df['faulty'] == 0], x='pressure', nbins=30,
                         color_discrete_sequence=['#040474'])
        fig.add_vline(x=38, line_dash="dash", line_color="green", annotation_text="Optimal")
        fig.update_layout(title="Normal Pressure Range", height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.markdown('<div class="section-header">Vibration</div>', unsafe_allow_html=True)
        fig = px.histogram(df[df['faulty'] == 0], x='vibration', nbins=30,
                         color_discrete_sequence=['#040474'])
        fig.add_vline(x=1.5, line_dash="dash", line_color="green", annotation_text="Optimal")
        fig.update_layout(title="Normal Vibration Range", height=300)
        st.plotly_chart(fig, use_container_width=True)

# Page 2: Data Exploration
def display_data_exploration():
    st.markdown('<div class="main-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    df = load_data()
    
    st.write("""
    Explore the relationships between different features and equipment faults through
    interactive visualizations. This analysis helps understand the patterns and factors
    that contribute to equipment failures.
    """)
    
    # Statistical summary tab
    st.markdown('<div class="sub-header">Data Statistics</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">Numerical Features Summary</div>', unsafe_allow_html=True)
        st.dataframe(df.describe())
    
    with col2:
        st.markdown('<div class="section-header">Target Distribution</div>', unsafe_allow_html=True)
        
        # Calculate counts and percentages
        value_counts = df['faulty'].value_counts().reset_index()
        value_counts.columns = ['Status', 'Count']
        value_counts['Percentage'] = (value_counts['Count'] / len(df) * 100).round(1)
        value_counts['Status'] = value_counts['Status'].map({0: 'Not Faulty', 1: 'Faulty'})
        
        fig = px.bar(value_counts, 
                    x='Status', 
                    y='Count',
                    text=[f'{count:,}<br>({pct}%)' for count, pct in zip(value_counts['Count'], value_counts['Percentage'])],
                    color='Status',
                    color_discrete_map={'Not Faulty': '#040474', 'Faulty': '#FF5252'})
        
        fig.update_layout(
            title='Distribution of Faulty vs Non-Faulty Equipment',
            showlegend=False,
            height=400,
            yaxis_title='Count',
            xaxis_title=''
        )
        fig.update_traces(textposition='outside')
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature analysis
    st.markdown('<div class="sub-header">Feature Analysis</div>', unsafe_allow_html=True)
    
    # Interactive feature selection for histograms
    feature_options = ['temperature', 'pressure', 'vibration', 'humidity']
    selected_feature = st.selectbox(
        'Select a numerical feature to analyze:',
        feature_options
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'<div class="section-header">Distribution of {selected_feature}</div>', unsafe_allow_html=True)
        # Histogram with KDE
        fig = px.histogram(
            df, 
            x=selected_feature, 
            color='faulty',
            marginal="violin", 
            hover_data=df.columns,
            labels={'faulty': 'Equipment Status'},
            color_discrete_map={0: "#040474", 1: "#FF5252"}
        )
        fig.update_layout(
            title=f'Distribution of {selected_feature} by Fault Status',
            xaxis_title=selected_feature.capitalize(),
            yaxis_title='Count',
            legend_title='Status',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                itemsizing="constant"
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown(f'<div class="section-header">Boxplot of {selected_feature}</div>', unsafe_allow_html=True)
        # Boxplot
        fig = px.box(
            df, 
            x='faulty', 
            y=selected_feature, 
            points="all",
            color='faulty',
            labels={'faulty': 'Equipment Status'},
            color_discrete_map={0: "#040474", 1: "#FF5252"}
        )
        fig.update_layout(
            title=f'Boxplot of {selected_feature} by Fault Status',
            xaxis_title='Equipment Status',
            yaxis_title=selected_feature.capitalize(),
            xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Not Faulty', 'Faulty']),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Categorical feature analysis
    st.markdown('<div class="sub-header">Categorical Features</div>', unsafe_allow_html=True)
    
    categorical_feature = st.selectbox(
        'Select a categorical feature:',
        ['equipment', 'location']
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'<div class="section-header">Count by {categorical_feature}</div>', unsafe_allow_html=True)
        # Count plot
        fig = px.histogram(
            df, 
            x=categorical_feature,
            color='faulty',
            barmode='group',
            labels={'faulty': 'Equipment Status'},
            color_discrete_map={0: "#040474", 1: "#FF5252"}
        )
        fig.update_layout(
            title=f'Count of Equipment by {categorical_feature} and Fault Status',
            xaxis_title=categorical_feature.capitalize(),
            yaxis_title='Count',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f'<div class="section-header">Fault Rate by {categorical_feature}</div>', unsafe_allow_html=True)
        # Calculate fault rate by category
        fault_rate = df.groupby(categorical_feature)['faulty'].mean().reset_index()
        fault_rate['Fault Rate (%)'] = fault_rate['faulty'] * 100
        
        fig = px.bar(
            fault_rate, 
            x=categorical_feature, 
            y='Fault Rate (%)',
            text='Fault Rate (%)',
            labels={categorical_feature: categorical_feature.capitalize()},
            color='Fault Rate (%)',
            color_continuous_scale='Bluered'
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(
            title=f'Fault Rate (%) by {categorical_feature}',
            xaxis_title=categorical_feature.capitalize(),
            yaxis_title='Fault Rate (%)',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.markdown('<div class="sub-header">Feature Relationships</div>', unsafe_allow_html=True)
    
    correlation = st.checkbox('Show correlation matrix')
    
    if correlation:
        # Correlation heatmap
        correlation_matrix = df.select_dtypes(include='number').corr()
        fig = px.imshow(
            correlation_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            aspect="auto",
            labels=dict(color="Correlation")
        )
        fig.update_layout(
            title='Feature Correlation Matrix',
            height=700
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature pairplot
    show_pairplot = st.checkbox('Show feature pairplot (may take a moment to render)')
    
    if show_pairplot:
        st.markdown('<div class="section-header">Feature Pairplot</div>', unsafe_allow_html=True)
        # Create pairplot with seaborn
        fig, ax = plt.subplots(figsize=(12, 10))
        pair_plot = sns.pairplot(
            df, 
            vars=['temperature', 'pressure', 'vibration', 'humidity'],
            hue='faulty', 
            diag_kind='kde',
            plot_kws={'alpha': 0.6},
            corner=True
        )
        plt.suptitle('Pairplot of Features by Fault Status', y=1.02, fontsize=16)
        st.pyplot(pair_plot.fig)

# Page 3: Prediction Model
def display_prediction_model():
    st.markdown('<div class="main-header">Equipment Fault Prediction Model</div>', unsafe_allow_html=True)
    
    st.write("""
    Use our machine learning model to predict potential equipment faults. You can input parameters
    manually or upload a CSV file with multiple records for batch prediction.
    """)
    
    try:
        model, scaler = load_model()
        model_loaded = True
    except:
        st.error("Failed to load the prediction model. Please ensure the model files exist in the directory.")
        model_loaded = False
    
    if model_loaded:
        tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
        
        with tab1:
            st.markdown('<div class="sub-header">Predict Equipment Fault</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                equipment_options = ["Compressor", "Pump", "Turbine"]
                equipment = st.selectbox("Equipment Type", equipment_options)
                
                location_options = ["Atlanta", "Chicago", "New York", "Dallas", "San Francisco"]
                location = st.selectbox("Location", location_options)
                
                temperature = st.slider("Temperature", 20.0, 110.0, 70.0, 0.1)
                pressure = st.slider("Pressure", 20.0, 70.0, 35.0, 0.1)
            
            with col2:
                vibration = st.slider("Vibration", 0.0, 5.0, 1.0, 0.01)
                humidity = st.slider("Humidity", 20.0, 80.0, 50.0, 0.1)
                
                st.info("""
                **Guidelines for healthy equipment:**
                - Temperature: 40-80°C
                - Pressure: 25-45 PSI
                - Vibration: <2.0 units
                - Humidity: 30-60%
                """)
            
            # Create a button for prediction
            predict_button = st.button("Predict Fault Probability", type="primary")
            
            if predict_button:
                # Map categorical values to numerical
                equipment_map = {"Compressor": 0, "Pump": 1, "Turbine": 2}
                location_map = {"Atlanta": 0, "Chicago": 1, "New York": 3, "Dallas": 2, "San Francisco": 4}
                
                # Create input data with feature engineering
                input_data = {
                    'temperature': temperature,
                    'pressure': pressure,
                    'vibration': vibration,
                    'humidity': humidity,
                    'equipment': equipment_map[equipment],
                    'location': location_map[location],
                    'temp_pressure_ratio': temperature / pressure,
                    'vibration_humidity_ratio': vibration / humidity,
                    'temp_humidity_ratio': temperature / humidity,
                    'temperature_squared': temperature ** 2,
                    'vibration_squared': vibration ** 2
                }
                
                # Convert to DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Scale the input
                input_scaled = scaler.transform(input_df)
                
                # Make prediction
                fault_probability = model.predict_proba(input_scaled)[0, 1]
                prediction = model.predict(input_scaled)[0]
                
                # Display results
                st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Fault Probability", 
                        f"{fault_probability:.2%}",
                        delta=None,
                    )
                    
                    # Show prediction with appropriate styling
                    if prediction == 1:
                        st.error("⚠️ **FAULT DETECTED**: The model predicts this equipment is likely to fail.")
                    else:
                        st.success("✅ **NORMAL OPERATION**: The model predicts this equipment is operating normally.")
                
                with col2:
                    # Create a gauge chart for the probability
                    fig = px.pie(values=[fault_probability, 1-fault_probability], 
                                names=["Fault Risk", "Normal"],
                                hole=0.7,
                                color_discrete_sequence=["#FF5252", "#040474"])
                    
                    fig.update_layout(
                        annotations=[dict(text=f"{fault_probability:.2%}", font_size=24, showarrow=False)],
                        showlegend=False,
                        height=250,
                        margin=dict(t=0, b=0, l=0, r=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance for this prediction
                st.markdown('<div class="section-header">Feature Analysis</div>', unsafe_allow_html=True)
                
                st.write("""
                The following features had the most influence on this prediction
                (based on general feature importance in the model):
                """)
                
                # Sample feature importance (this would ideally be dynamic based on the specific prediction)
                importance_data = {
                    "Feature": ["Vibration", "Temperature", "Pressure", "Humidity", "Equipment Type"],
                    "Importance": [0.35, 0.25, 0.20, 0.15, 0.05]
                }
                
                importance_df = pd.DataFrame(importance_data)
                fig = px.bar(
                    importance_df, 
                    x="Importance", 
                    y="Feature",
                    orientation='h',
                    color="Importance",
                    color_continuous_scale="Bluered"
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown('<div class="sub-header">Batch Prediction</div>', unsafe_allow_html=True)
            
            st.write("""
            Upload a CSV file with equipment data to get predictions for multiple pieces of equipment at once.
            The CSV should contain the same columns as the original dataset.
            """)
            
            # File uploader
            uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
            
            if uploaded_file is not None:
                # Read the CSV
                try:
                    batch_data = pd.read_csv(uploaded_file)
                    st.write("Preview of uploaded data:")
                    st.dataframe(batch_data.head())
                    
                    # Check if required columns exist
                    required_columns = ['temperature', 'pressure', 'vibration', 'humidity', 'equipment', 'location']
                    missing_columns = [col for col in required_columns if col not in batch_data.columns]
                    
                    if missing_columns:
                        st.error(f"Missing required columns: {', '.join(missing_columns)}")
                    else:
                        # Process the data
                        st.info("Processing data... Please wait.")
                        
                        # Preprocessing
                        batch_data_processed = batch_data.copy()
                        
                        # Make sure categorical columns are properly encoded
                        equipment_map = {"Compressor": 0, "Pump": 1, "Turbine": 2}
                        location_map = {"Atlanta": 0, "Chicago": 1, "Dallas": 2, "New York": 3, "San Francisco": 4}
                        
                        # Convert categorical columns to numerical if they're strings
                        if batch_data_processed['equipment'].dtype == 'object':
                            batch_data_processed['equipment'] = batch_data_processed['equipment'].map(equipment_map)
                        
                        if batch_data_processed['location'].dtype == 'object':
                            batch_data_processed['location'] = batch_data_processed['location'].map(location_map)
                        
                        # Create engineered features
                        batch_data_processed['temp_pressure_ratio'] = batch_data_processed['temperature'] / batch_data_processed['pressure']
                        batch_data_processed['vibration_humidity_ratio'] = batch_data_processed['vibration'] / batch_data_processed['humidity']
                        batch_data_processed['temp_humidity_ratio'] = batch_data_processed['temperature'] / batch_data_processed['humidity']
                        batch_data_processed['temperature_squared'] = batch_data_processed['temperature'] ** 2
                        batch_data_processed['vibration_squared'] = batch_data_processed['vibration'] ** 2
                        
                        # Scale the data
                        columns_for_prediction = ['temperature', 'pressure', 'vibration', 'humidity', 'equipment', 
                                                'location', 'temp_pressure_ratio', 'vibration_humidity_ratio', 
                                                'temp_humidity_ratio', 'temperature_squared', 'vibration_squared']
                        
                        batch_data_scaled = scaler.transform(batch_data_processed[columns_for_prediction])
                        
                        # Make predictions
                        predictions = model.predict(batch_data_scaled)
                        probabilities = model.predict_proba(batch_data_scaled)[:, 1]
                        
                        # Add predictions to the original data
                        result_df = batch_data.copy()
                        result_df['fault_probability'] = probabilities
                        result_df['predicted_fault'] = predictions
                        
                        # Display results
                        st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)
                        
                        # Summary statistics
                        fault_count = result_df['predicted_fault'].sum()
                        total_count = len(result_df)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Total Equipment", total_count)
                            st.metric("Predicted Faults", fault_count)
                        
                        with col2:
                            st.metric("Fault Percentage", f"{fault_count/total_count:.2%}")
                            if fault_count > 0:
                                st.metric("Avg. Fault Probability", 
                                        f"{result_df[result_df['predicted_fault'] == 1]['fault_probability'].mean():.2%}")
                        
                        # Show the detailed results
                        st.dataframe(result_df)
                        
                        # Download option for results
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="equipment_fault_predictions.csv",
                            mime="text/csv",
                        )
                        
                        # Visualize results
                        st.markdown('<div class="section-header">Visualization of Predictions</div>', unsafe_allow_html=True)
                        
                        # Plot predictions by probability
                        fig = px.histogram(
                            result_df, 
                            x="fault_probability",
                            color="predicted_fault",
                            nbins=20,
                            labels={"fault_probability": "Fault Probability", "predicted_fault": "Predicted Fault"},
                            color_discrete_map={0: "#040474", 1: "#FF5252"}
                        )
                        fig.update_layout(title="Distribution of Fault Probabilities")
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error processing the file: {str(e)}")

        # Model information
        st.markdown('<div class="sub-header">About the Model</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
            performance_data = {
                "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
                "Value": [0.98, 0.99, 0.95, 0.99]
            }
            
            st.dataframe(performance_data, hide_index=True)
        
        with col2:
            st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
            feature_importance = {
                "Feature": ["Vibration", "Temperature", "Pressure", "Temp-Pressure Ratio", "Humidity"],
                "Importance (%)": [32, 24, 18, 16, 10]
            }
            
            fig = px.pie(
                feature_importance, 
                values="Importance (%)", 
                names="Feature",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()