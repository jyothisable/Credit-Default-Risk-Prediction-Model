# Features
categorical_ordinal_features = ['term', 'grade','sub_grade','emp_length', 'verification_status']

term_order = [' 36 months', ' 60 months']
grade_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
sub_grade_order = [grade + str(i) for grade in grade_order for i in range(1,6)]
emp_length_order = ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']
verification_status_order = ['Not Verified', 'Verified', 'Source Verified']

order_matrix = [term_order, grade_order, sub_grade_order, emp_length_order, verification_status_order]

categorical_nominal_features = ['home_ownership','purpose','title','initial_list_status','application_type'] # OHE with 1% threshold to be done

numerical_features = ['loan_amnt','revol_util']

numerical_skewed_features = ['annual_inc','dti','open_acc', 'pub_rec','revol_bal', 'total_acc', 'mort_acc','pub_rec_bankruptcies']

post_FE_features = ['annual_inc', 'dti', 'open_acc', 'mort_acc', 'loan_amnt', 'revol_util',
       'term', 'grade', 'sub_grade', 'verification_status',
       'home_ownership_mortgage', 'home_ownership_rent', 'purpose_car',
       'purpose_credit_card', 'purpose_debt_consolidation',
       'purpose_home_improvement', 'purpose_major_purchase',
       'purpose_small_business', 'title_consolidation',
       'title_debt consolidation', 'title_other', 'title_infrequent_sklearn',
       'zipcode_00813', 'zipcode_05113', 'zipcode_11650', 'zipcode_29597',
       'zipcode_86630', 'zipcode_93700', 'age_of_credit_4',
       'age_of_credit_10']