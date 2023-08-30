| Column Name                         | Description                                                    |
|------------------                   |----------------------------------------------------------------|
| `Item_Identifier`                   | Unique product ID                                              |
| `Item_Weight`                       | Weight of product                                              |
| `Item_Fat_Content`                  | Checks the Concentration of fat in the product                 |
| `Item_Visibility`                   | The % of total display area of all similar products in a store |
| `Item_Type`                         | Product Category                                               |
| `Item_MRP`                          | Maximum Retail Price for a Product                             |
| `Outlet_Identifier`                 | Store ID                                                       |
| `Outlet_Establishment_Year`         | The year in which store was established                        |
| `Outlet_Size`                       | The size of the store (Area Size Category)                     |
| `Outlet_Location_Type`              | In Terms of city Tiers (Size)                                  |
| `Outlet_Type`                       | Grocery store or a type of    supermarket                      |
| `Item_Outlet_Sales`                 | Sales of the product In the Specific outlet                    |


-----
# Problem Statement:
The data scientists at BigMart have collected sales data for 1559 products across 10 stores in different cities for the year 2013. Now each product has certain attributes that sets it apart from other products.

### Breakdown of the Problem Statement:
* Supervised machine learning problem.
* The target value will be `Item_Outlet_Sales`.

### Aim of the NoteBook:
The objective is to create a model that can predict the sales per product for each store. Using this model, BigMart will try to understand the properties of products and stores which play a key role in increasing sales.


## **Reference:**

<a id='section-0'></a>
<h2 style="color:darkolivegreen;">Table of Contents:</h2>
<ol>
    <li><a href="#section-1" style="color:#0000ff;">Problem Statement</a></li>
    <li><a href="#section-2" style="color:#0000ff;">Libraries </a></li>
    <li><a href="#section-3" style="color:#0000ff;">File Paths </a></li>
    <li><a href="#section-4" style="color:#0000ff;">UNDERSTANDING - DATA  </a></li>
    <li><a href="#section-5" style="color:#0000ff;">Data Preprocessing</a></li>
    <li><a href="#section-7" style="color:#0000ff;">Exploratory Data Analysis (EDA) </a></li>
      <ol>
        <li><a href="#section-7.0" style="color:#0000ff;">Univariate Plots </a></li>
        <ol>
          <li><a href="#section-7.0.0" style="color:#0000ff;">Categorical columns </a></li>
          <ol>
            <li><a href="#section-7.0.0.1" style="color:#0000ff;">Analysis of Item_Identifier</a></li>
            <li><a href="#section-7.0.0.2" style="color:#0000ff;">Analysis of Item Fat Content </a></li>
            <li><a href="#section-7.0.0.3" style="color:#0000ff;">Analysis of Item_Type </a></li>
            <li><a href="#section-7.0.0.4" style="color:#0000ff;">Analysis of Outlet_Identifier </a></li>
            <li><a href="#section-7.0.0.5" style="color:#0000ff;">Analysis of Outlet_Size </a></li>
            <li><a href="#section-7.0.0.6" style="color:#0000ff;">Analysis of Outlet_Location_Type </a></li>
            <li><a href="#section-7.0.0.7" style="color:#0000ff;">Analysis of Outlet_Type </a></li>
          </ol>
        </ol>
        <ol>
          <li><a href="#section-7.0.1" style="color:#0000ff;">Numerical columns </a></li>
            <ol>
              <li><a href="#section-7.0.1.1" style="color:#0000ff;">Analysis of Item Weights </a></li>
              <li><a href="#section-7.0.1.2" style="color:#0000ff;">Analysis of Item Visibility </a></li>
              <li><a href="#section-7.0.1.3" style="color:#0000ff;">Analysis of Item_MRP </a></li>
              <li><a href="#section-7.0.1.4" style="color:#0000ff;">Analysis of Outlet_Age </a></li>
              <li><a href="#section-7.0.1.5" style="color:#0000ff;">Analysis of Item_Outlet_Sales </a></li>
              <li><a href="#section-7.0.1.6" style="color:#0000ff;">Analysis of Outlet Sales </a></li>
              <li><a href="#section-7.0.1.7" style="color:#0000ff;">Analysis of Item Weight and Item Outlet Sales </a></li>
            </ol>
        </ol>
        <li><a href="#section-7.1" style="color:#0000ff;">Bivariate Analysis </a></li>
        <ol>
          <li><a href="#section-7.1.0" style="color:#0000ff;">Analysis of Sales per item type </a></li>
          <li><a href="#section-7.1.1" style="color:#0000ff;">Analysis of Sales per outlet </a></li>
          <li><a href="#section-7.1.2" style="color:#0000ff;">Analysis of Sales per outlet type </a></li>
          <li><a href="#section-7.1.3" style="color:#0000ff;">Analysis of Sales per outlet size </a></li>
          <li><a href="#section-7.1.4" style="color:#0000ff;">Analysis of Sales per location type </a></li>
          <li><a href="#section-7.1.5" style="color:#0000ff;">
          Realizations </a></li>
        </ol>
        <li><a href="#section-7.2" style="color:#0000ff;">Some other Bivariate Analysis </a></li>
        <ol>
            <li><a href="#section-7.2.1" style="color:#0000ff;">Analysis of Item Visibility and Outlet Size </a></li>
            <li><a href="#section-7.2.2" style="color:#0000ff;">Analysis of Item Visibility and Outlet Type </a></li>
            <li><a href="#section-7.2.3" style="color:#0000ff;">Analysis of Outlet Establishment Year and Outlet Type </a></li>
            <li><a href="#section-7.2.4" style="color:#0000ff;">Analysis of Item Fat Content and Item Outlet Sales </a></li>
            <li><a href="#section-7.2.5" style="color:#0000ff;">Analysis of Outlet Establishment Year and Outlet Location Type </a></li>
            <li><a href="#section-7.2.6" style="color:#0000ff;">Analysis of Outlet Establishment Year and Item Outlet Sales </a></li>
            <li><a href="#section-7.2.7" style="color:#0000ff;">Analysis of Outlet Size and Outlet Type </a></li>
            <li><a href="#section-7.2.8" style="color:#0000ff;">Analysis of Outlet Size and Outlet Location Type </a></li>
            <li><a href="#section-7.2.9" style="color:#0000ff;">Analysis of Outlet Location Type and Outlet Type </a></li>
        </ol>
      </ol>
    <li><a href="#section-8" style="color:#0000ff;">Feature Engineering </a></li>
    <li><a href="#section-9" style="color:#0000ff;">Splitting Dataset into `X` and `Y` variables </a></li>
    <li><a href="#section-10" style="color:#0000ff;">Selecting model </a></li>
      <ol>
        <li><a href="#section-10.0" style="color:#0000ff;">Linear Regression </a></li>
        <li><a href="#section-10.1" style="color:#0000ff;">Lasso Regressor </a></li>
        <li><a href="#section-10.2" style="color:#0000ff;">Ridge Regression </a></li>
        <li><a href="#section-10.3" style="color:#0000ff;">Random Forest Regressor </a></li>
        <li><a href="#section-10.4" style="color:#0000ff;">XGB </a></li>
      </ol>
    <li><a href="#section-11" style="color:#0000ff;">Conclusion </a></li>
    <li><a href="#section-12" style="color:#0000ff;">Realizations </a></li>
</ol>
