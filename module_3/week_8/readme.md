These standard docs collectively support the lab deliverables this week: compare feature
sets, measure performance, training/validation stability, and evaluate feature importance
and PCA.
 https://scikit-learn.org/stable/modules/feature_selection.html
 https://scikit-learn.org/stable/modules/permutation_importance.html
 https://scikit-learn.org/stable/modules/cross_validation.html
 https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

========

Page 1 of
1
Scikit-learn directly documents filter methods, model-based embedded selection, recursive
elimination, and sequential forward/backward selection with cross-validation.
 https://scikit-learn.org/stable/modules/feature_selection.html
 https://scikit-learn.org/stable/modules/generated/
sklearn.feature_selection.SequentialFeatureSelector.html


=============


Page 1 of
1
PCA docs cover explained variance and transformation; the unsupervised reduction guide
helps frame when dimensionality reduction is useful; TensorFlow autoencoders provide the
conceptual nonlinear DR reference.
 https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
 https://scikit-learn.org/stable/modules/unsupervised_reduction.html
 https://www.tensorflow.org/tutorials/generative/autoencoder

=================



  
Activity 1: Compare wrapper, filter, and embedded methods on one dataset
Page 1 of
1
These examples compare sequential/model-based selection and univariate filtering in code.
 https://scikit-learn.org/stable/auto_examples/feature_selection/
plot_select_from_model_diabetes.html
 https://scikit-learn.org/stable/auto_examples/feature_selection/plot_feature_selection.html