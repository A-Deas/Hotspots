$\textbf{\Large 1) Data preparation}$

$\quad$ \2020 USA County Shapefile contains the cleaned and matched (to the data) shapefile needed to print the maps and construct the Q matrices. This shapefile is FIPS_usa.shp, which was constructed from the original shapefile cb_2020_us_county_20m.shp obtained from the Census Bureau's website. However, the 2020 shapefile currently on their website has been updated, thus this outdated original has been included for reference.

$\quad$ \Dirty Data contains the raw biennial SVI data and centers of population from the Census Bureau.

$\quad$ \Clean Data contains the cleaned and curated data files used in the analysis. The OD (mortality) and DR (dispensing) rates were cleaned in 
excel separately, however, the python program used for cleaning the SVI data is included ('Clean: SVI Data.py'). Moreover, this program details the 
county structure changes from 2014 to 2020 in the form of those counties which needed to be dropped from, or added to, the datasets in 2020. 
Even though the paper only showcased the results for the mortality rates, dispensing rates and disability rates, all SVI variables except 
'Housing Cost Burden' and 'No Health Insurance' are included in this folder. The latter two categories were not available across all years in our study, 
so they were dropped.

$\textbf{\Large 2) Running the Kalman filter}$

$\quad$ The Kalman filter ('Kalman filter: Predictive.py') utilizes covariance matrices that are constructed in the program 'Q Matrices Construction.py' 
and then stored in \Covariance Matrices. The size of these matrices exceeds 100 MB, therefore we were not able to upload them to this Github repo. 
Thus, in order to reproduce the results, one will need to run 'Q Matrices Construction.py' first to construct the required matrices. 
In this program, one will find the details on the specific values used to construct these matrices which are designed to capture the underlying 
spatial dimension of our data.

$\quad$ Once the matrices are constructed, one can then run the Kalman filter which utilizes the data from 2014 to 2019 to learn, then makes autonomous 
predictions for 2020, as described in the paper. These predictions are stored in \Kalman Predictions.

$\textbf{\Large 3) Unpacking the performance of the Kalman filter}$

$\quad$ With the Kalman estimates and predictions in hand, one can then construct the maps and histograms showcased in the paper for every year in 
the study periodn and every variable that was run through the Kalman filter (not limited to the three variables explicitly analyzed in the paper: 
OD, DR and SVI Disability).

$\bullet\textbf{Heat Maps:}$ These maps visualize the national vulnerability profiles for each variable. In each year, the Kalman filter's posterior distribution
is approximated by fitting a normal distribution to that year's estimates (2019 or prior) or predictions (2020). Then utilizing this fitted normal 
distribution, we compute the cdf values for each county and categorize them into 20 different heat levels going up by 5th percentiles.

$\bullet\textbf{Hotspot maps:}$ These maps hone in on the annual hotspots, those counties whose rate exceeds the 95th percentile (highest vulnerability level).

$\bullet\textbf{Error Histograms:}$ These histograms detail the yearly distribution of errors in the Kalman filter estimates. The error is computed for each 
county as the absolute difference of the Kalman estimate and the actual data value.

$\bullet\textbf{Accuracy maps:}$ These maps utilize accuracy levels to visualize the accuracy of the Kalman filter's estimates. Accuracy is measured for each county 
as that county's error divided by the maximum annual error.

$\bullet\textbf{Hotspot Accuracy Maps:}$ These maps detail the accuracy of the filter in predicting the hotspots, measured as the number of hotspots correctly
predicted divided by the total number of hotspots in the data.

$\bullet\textbf{Efficacy Computations:}$ This program simply prints (without the corresponding maps) the results of the efficacy metrics used in the paper: 
average error, maximum error, general accuracy and hotspot accuracy.

$\textbf{\Large 4) Kalman filter trained on less data}$

$\quad$ Here ToLD is an abbreviation for 'Trained on Less Data'.

$\quad$ This folder contains the results of our investigation into training the Kalman filter on less data and forecasting for multiple years. The filter ToLD
still utilizes the same covariance matrices, so these are pulled from the original \Covariance Matrices folder. In the 'Kalman Filter ToLD.py', we iteratively
train the filter on four years of data down to just a single year, then predict for the corresponding difference in time. For example, if the filter is 
trained on only a single year, it is initialized with the 2014 data, learns in 2015 then makes autonomous predictions from 2016 to 2020. The results are 
stored in \Kalman Filter ToLD\Kalman Predictions ToLD. One can then construct the corresponding accuracy maps, error histograms and print the efficacy metric 
results in the same fashion as is done and described for the fully trained filter.

$\textbf{\Large 5) Accessing the supplemental images}$

$\quad$ The supplemental maps and histograms for the entire study period have been previously printed and stored in the \Images folder, one does not 
have to run all of the programs to view these.
