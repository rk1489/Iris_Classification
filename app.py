import streamlit as st
import pickle



knn=pickle.load(open('knn.pkl','rb'))
svm=pickle.load(open('svm.pkl','rb'))
tree=pickle.load(open('tree.pkl','rb'))
rf=pickle.load(open('rf.pkl','rb'))

def classify(x):
    if len(x) == 0:
        return 'Unidentified Species'
    else:
        return x
    
def main():
    st.title("Customer Classifier ML App")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Customer Classification</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['K - Nearest Neighbour (KNN)','Support Vector Classifier (SVC)','Decision Tree', 'Random Forest']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)
    
    sepallength=st.slider('Select Sepal Length', 0.0, 10.0)
    sepalwidth=st.slider('Select Sepal Width', 0.0, 10.0)
    petallength=st.slider('Select Petal Length', 0.0, 10.0)
    petalwidth=st.slider('Select Petal Width', 0.0, 10.0)
    
    inputs=[[sepallength,sepalwidth,petallength,petalwidth]]
    
    if st.button('Classify'):
        if option=='K - Nearest Neighbour (KNN)':
            st.success(classify(knn.predict(inputs)))
        elif option=='Support Vector Classifier (SVC)':
            st.success(classify(svm.predict(inputs)))
        elif option=='Decision Tree':
            st.success(classify(tree.predict(inputs)))
        elif option=='Random Forest':
            st.success(classify(rf.predict(inputs)))
            

if __name__=='__main__':
    main()
