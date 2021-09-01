from CNSVM import CNSVM_Model

if __name__ == "__main__":
    
    # Step 1 : initialize a CNSVM class
    
    example_cnsvm = CNSVM_Model()
    
    # Step 2 : load and prepare data for training of CNSVM model
    
    example_cnsvm.prepare_training_data(
        file_name = "training_data.csv", # define the loaction of the .csv file containing training data
        sep = ";" # specify how the .csv file is delimited
    )
    
    # Step 3 : define and fit neural network portion of the model
    
    example_cnsvm.define_model(
        select_act = "swish", # select between swish or relu activation
        opt = "sgd" # select between adam or stochastic gradient descent optimization
    )
    example_cnsvm.fit_CNN_model(
        use_gpu = False # select True if the user wishes to use GPU (default is True)
    )
    
    # Step 4 : once trained extract the base neural network
    
    example_cnsvm.get_base_nn()
    
    # Step 5 : tune the Support Vector Machine activation layer and fit the tuned layer to the model, thus creating the final CNSVM model
    
    example_cnsvm.tune_SVM()
    example_cnsvm.fit_SVM()
    
    # Step 6 : evaluate the performance of the final CNSVM on the test set
    
    example_cnsvm.evaluate_CNSVM()
    
    # Step 7 : save the CNSVM model files
    
    example_cnsvm.save_model(
        "example_cnsvm" # specify a base file name to save the neural network and SVM activation layer to
    )
    
    # Step 8 : make predictions using the CNSVM model
    
    # create an example hyperspectral signature to test the algorithm works
    # example of a Basal Cell Carcinoma patient
    
    import numpy as np
    example_signature = np.array([16.98339653, 17.93694115, 18.57286644, 19.07141685, 19.05452347,
                                    20.99124336, 21.15433311, 21.5257473 , 21.57803917, 22.55696678,
                                    24.04073334, 23.77472687, 22.17740822, 23.57593155, 23.67067719,
                                    23.94239426, 23.78603554, 24.55370522, 23.79878044, 25.87969208,
                                    24.57605743, 25.83368301, 25.28944588, 25.70561028, 26.23495865,
                                    26.15563202, 26.53599548, 26.58680344, 27.16779709, 26.99596596,
                                    27.26475906, 27.17713547, 27.2520771 , 27.17879868, 28.68021011,
                                    28.88275719, 27.93162346, 28.55453682, 29.02724648, 28.72593117,
                                    28.43540001, 29.51581192, 29.56198692, 29.33459091, 30.39468193,
                                    30.76584244, 29.79483223, 30.77136993, 29.9730835 , 29.57049751,
                                    31.12574196, 31.6844101 , 31.32255745, 31.88443756, 31.24224854,
                                    31.90246201, 32.74693298, 32.29421997, 33.34341431, 32.9422226 ,
                                    32.51787186, 32.14392853, 33.50373459, 32.91928101, 33.19164276,
                                    34.49824524, 33.96783066, 32.91753006, 34.52427673, 34.30975723,
                                    34.71454239, 34.03808212, 34.37699509, 34.23149109, 34.01000595,
                                    34.05374527, 33.21917343, 34.29608917, 34.74617767, 33.71575165,
                                    34.15397263, 34.37569427, 34.94247437, 35.08311081, 34.22791672,
                                    35.09651184, 34.77617264, 34.5438118 , 35.39922714, 35.81271362,
                                    35.39262772, 35.35317612, 34.93600464, 35.04853439])
    
    # predict the label
    
    label = example_cnsvm.predict_label(example_signature)
    
    # predict the probability of label association
    
    probabilities = example_cnsvm.predict_prob(example_signature)
    
    # print to console the results
    
    print(f"class label = {label[0]}")
    print(f"Probability of healthy skin = {probabilities[:,0] * 100}%")
    print(f"Probability of cancer = {probabilities[:,1] * 100}%")


"""
If the user already has a pretrained model, then steps 1 to 8 can be substituted by:

example_cnsvm = CNSVM()
example_cnsvm.load_model(
    "example_cnsvm_Neural_Net.h5", # loaction of the .h5 file with network weights
    "example_cnsvm_SVM_Actiavtion_Layer.joblib" # loaction of the .joblib file with the SVM activation layer
)

"""

