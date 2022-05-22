import streamlit as st
import pickle
import lightgbm
from sklearn.metrics import classification_report,plot_precision_recall_curve,plot_confusion_matrix,precision_recall_fscore_support,plot_roc_curve

def app():
    with st.sidebar:
        st.title('Stroke Prediction using Machine Learning')

        st.write('This model which predicts whether a patient is likely to get a stroke based on the parameters like gender, age various diseases and smoking status.')
        st.markdown('_For Machine Learning - 19CS601_')
    
    st.title('Model Overview')
    st.write('The model performance of the dataset is presented below.')

    # Retreving model and it's components for performance metric
    model = pickle.load(open("apps\models\gbm\gbm-model-pickle.sav", 'rb'))
    X_test = pickle.load(open("apps\models\gbm\gbm-xtest.sav", 'rb'))
    Y_test = pickle.load(open("apps\models\gbm\gbm-ytest.sav", 'rb'))
    Y_pred = model.predict(X_test)

    st.header('Model performance')
    #result = model.score(X_test, Y_test)

    precision,recall,f1_sc,support=precision_recall_fscore_support(Y_test,Y_pred)
    accuracy=model.score(X_test,Y_test)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", round(accuracy,4), "")
    col2.metric("Recall", round(recall[0],4), "")
    col3.metric("F-measure", round(f1_sc[0],4), "")
    col4.metric("Support", support[0], "")

    st.subheader("Model type: ")
    st.write(model)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.subheader("Confusion Matrix: ")
    plot_confusion_matrix(model, X_test, Y_test, display_labels=['NoStroke','Stroke'])
    st.pyplot()
    #st.table(confusion_matrix(Y_test, Y_pred))

    st.subheader("ROC Curve")
    plot_roc_curve(model, X_test, Y_test)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    st.subheader("Precision-Recall Curve")
    plot_precision_recall_curve(model, X_test, Y_test)
    st.pyplot()

    st.subheader('Other metrics:')
    report=classification_report(Y_test, Y_pred, target_names=None)
    st.code(report)