# Function to preprocess the ecological health dataset
    
def preprocess_ecological_dataset(filepath="ecological_health_dataset.csv"):
    """
    Load and preprocess the ecological health dataset
    
    Parameters:
    -----------
    filepath : str
        Path to the dataset CSV file
    
    Returns:
    --------
    X : numpy.ndarray
        Preprocessed feature matrix
    y : numpy.ndarray
        Biodiversity_Index target variable
    feature_names : list
        Names of the features after preprocessing
    """
    print("Loading and preprocessing the ecological health dataset...")
    
    # Load the CSV file
    df = pd.read_csv(filepath)
    
    # Displaying basic information
    print(f"Dataset shape: {df.shape}")
    print(f"Target variable distribution:\n{df['Biodiversity_Index'].value_counts().sort_index().head()}")
    
    # Remove  timestamp column if it exists
    if 'Timestamp' in df.columns:
        df = df.drop(columns=['Timestamp'])
    
    # Checking for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("Missing values detected. Filling with appropriate values...")
        # Filling numeric columns with median
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Filling categorical columns with mode
        categorical_cols = ['Pollution_Level', 'Ecological_Health_Label']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])
    
    # Identifying categorical columns for encoding
    categorical_cols = [col for col in ['Pollution_Level', 'Ecological_Health_Label'] if col in df.columns]
    
    # Creating preprocessor with column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), 
             [col for col in df.columns if col not in categorical_cols + ['Biodiversity_Index']]),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ],
        remainder='drop'
    )
    
    # Extracting features and target
    X = df.drop(columns=['Biodiversity_Index'])
    y = df['Biodiversity_Index'].values
    
    # Fitting and transforming the features
    X_processed = preprocessor.fit_transform(X)
    
    # feature names after preprocessing
    numeric_cols = [col for col in df.columns if col not in categorical_cols + ['Biodiversity_Index']]
    
    #  one-hot encoded feature names
    ohe = preprocessor.named_transformers_['cat']
    cat_features = []
    for i, col in enumerate(categorical_cols):
        categories = ohe.categories_[i][1:]  
        cat_features.extend([f"{col}_{cat}" for cat in categories])
    
    feature_names = numeric_cols + cat_features
    
    print(f"Processed features shape: {X_processed.shape}")
    print(f"Target variable shape: {y.shape}")
    
    return X_processed, y, feature_names