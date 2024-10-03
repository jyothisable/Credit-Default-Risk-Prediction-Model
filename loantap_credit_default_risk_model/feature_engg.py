def feature_evaluator(x_train, y_train, x_test, y_test, feature_engineering_pipeline,target_pipeline):
    ''' 
    simple helper function to grid search an ExtraTreesClassifier model and 
    print out a classification report for the best param set.
    Best here is defined as having the best cross-validated accuracy on the training set
    '''
    
    params = {  # some simple parameters to grid search
        'base_model__max_depth': [None],
        'base_model__n_estimators': [50],
        'base_model__criterion': ['gini'],
        }

    base_model = ExtraTreesClassifier(n_jobs=6,random_state=42) # this evaluates our feature engineering pipeline
    
    model_with_fe = Pipeline([
        ('feature_engineering_pipeline', feature_engineering_pipeline),
        ('base_model', base_model)
    ])
    

    model_grid_search = GridSearchCV(model_with_fe, param_grid=params, cv=3,n_jobs=6,verbose=True,scoring='f1')
    start_time = time.time()  # capture the start time

    parse_time = time.time()
    print(f"Parsing took {(parse_time - start_time):.2f} seconds")

    model_grid_search.fit(x_train,target_pipeline.fit_transform(y_train))
    fit_time = time.time()
    print(f"Training took {(fit_time - start_time):.2f} seconds")

    best_model = model_grid_search.best_estimator_

    y_pred=best_model.predict(x_test)
    print(classification_report(target_pipeline.transform(y_test), y_pred))
    
    end_time = time.time()
    print(f"Overall took {(end_time - start_time):.2f} seconds")
    
    return best_model