#load libraries
library(tidyverse)
library(ROCR)
library(text2vec)
library(tm)
library(SnowballC)
library(glmnet)
library(vip)
library(naivebayes)
library(ranger)
library(xgboost)
library(ROCR)
library(textdata)
library(tidytext)

library(randomForest)
library(gbm)

#load data files
setwd('C:/Users/vaish/OneDrive/Documents/Predictive Modelling Project/')
train_x <- read_csv("airbnb_train_x_2024.csv")
train_y <- read_csv("airbnb_train_y_2024.csv")
test_x <- read_csv("airbnb_test_x_2024.csv")

#join the training y to the training x file
#also turn the target variables into factors
train <- cbind(train_x, train_y) %>%
  mutate(perfect_rating_score = as.factor(perfect_rating_score),
         high_booking_rate = as.factor(high_booking_rate),
         state = as.factor(state)) 

train$host_since_numeric <- as.numeric(difftime(Sys.Date(), as.Date(train$host_since), units = "days"))
test_x$host_since_numeric <- as.numeric(difftime(Sys.Date(), as.Date(test_x$host_since), units = "days"))
# EXAMPLE PREDICTIONS FOR CONTEST 2

#create a simple model to predict high_booking_rate and generate predictions in the test data
# make sure there are no NAs in your predictions
train_rate <- train %>%
  select(-c(perfect_rating_score))

summary(train_rate)

city_freq <- table(train_rate$city)




#Cleaning data (host_total_listings_count, square_feet, availability_30, availability_60, availability_90, availability_365)
train_rate <- train_rate %>%
  mutate(
    host_since_numeric = as.numeric(difftime(Sys.Date(), as.Date(host_since), units = "days")),
    host_total_listings_count = ifelse(is.na(host_total_listings_count), 
                                       median(host_total_listings_count, na.rm=TRUE), host_total_listings_count),
    host_total_listings_count = ifelse(host_total_listings_count == 0, 1, host_total_listings_count),
    availability_365 = ifelse(availability_365 == 0, NA, availability_365),
    availability_365 = ifelse(is.na(availability_365), median(availability_365, na.rm=TRUE), availability_365),
    availability_30 = ifelse(availability_30 == 0, NA, availability_30),
    availability_30 = ifelse(is.na(availability_30), median(availability_30, na.rm=TRUE), availability_30),
    availability_60 = ifelse(availability_60 == 0, NA, availability_60),
    availability_60 = ifelse(is.na(availability_60), median(availability_60, na.rm=TRUE), availability_60),
    availability_90 = ifelse(availability_90 == 0, NA, availability_90),
    availability_90 = ifelse(is.na(availability_90), median(availability_90, na.rm=TRUE), availability_90),
    
    bathrooms = ifelse(is.na(bathrooms), median(bathrooms, na.rm = TRUE), bathrooms),
    guests_included=ifelse(is.na(guests_included), median(guests_included, na.rm = TRUE), guests_included),
    
    security_deposit = ifelse(is.na(security_deposit), median(security_deposit, na.rm = TRUE), security_deposit),
    host_acceptance_rate = parse_number(host_acceptance_rate),
    accommodates = ifelse(is.na(accommodates), mean(accommodates, na.rm=TRUE), accommodates),
    host_acceptance = as.factor(ifelse(is.na(host_acceptance_rate), "MISSING",
                                       ifelse(host_acceptance_rate == 100, "ALL", "SOME"))),
    host_response = as.factor(ifelse(is.na(host_response_rate), "MISSING",
                                     ifelse(host_response_rate == 100, "ALL",  "SOME"))),
    beds=ifelse(is.na(beds),mean(beds,na.rm=TRUE),beds),
    cleaning_fee=ifelse(is.na(cleaning_fee),0,cleaning_fee),
    bedrooms = ifelse(is.na(bedrooms), median(bedrooms, na.rm = TRUE), bedrooms),
    property_category = as.factor(case_when(
      property_type %in% c("Apartment", "Serviced apartment", "Loft") ~ "apartment",
      property_type %in% c("Bed & Breakfast", "Boutique hotel", "Hostel") ~ "hotel",
      property_type %in% c("Townhouse", "Condominium") ~ "condo",
      property_type %in% c("Bungalow", "House") ~ "house",
      TRUE ~ "other")),
    state = as.factor(case_when(
      state %in% c("ca", "Ca", "CA") ~ "CA",
      state %in% c("il", "IL") ~ "IL",
      state %in% c("ny", "NY") ~ "NY",
      state %in% c("CO") ~ "CO",
      state %in% c("LA") ~ "LA",
      state %in% c("MA") ~ "MA",
      state %in% c("MD") ~ "MD",
      state %in% c("OR") ~ "OR",
      state %in% c("TN") ~ "TN",
      state %in% c("TX") ~ "TX",
      state %in% c("WA") ~ "WA",
      TRUE ~ "MISSING"
    )),
    bed_type = as.factor(bed_type),
    room_type = as.factor(room_type),
    maximum_nights=ifelse(is.na(maximum_nights), mean(maximum_nights, na.rm = TRUE), maximum_nights),
    monthly_price= ifelse(is.na(monthly_price), mean(monthly_price, na.rm = TRUE), monthly_price),
    square_feet=ifelse(is.na(square_feet), mean(square_feet, na.rm = TRUE), square_feet),
    weekly_price=ifelse(is.na(weekly_price), mean(weekly_price, na.rm = TRUE), weekly_price),
    cancellation_policy = as.factor(ifelse(cancellation_policy %in% c("super_strict_60", "super_strict_30"), 
                                           "strict", cancellation_policy)),
    price = ifelse(is.na(price), mean(price, na.rm = TRUE), price),
    bed_category = as.factor(ifelse(bed_type == "Real Bed", "Bed", "Other")),
    latitude = ifelse(is.na(latitude), 0, latitude),
    longitude = ifelse(is.na(longitude), 0, longitude),
    country=ifelse(is.na(country), "United States", country),
    country_code= ifelse(is.na(country_code), "US", country_code),
    host_location = ifelse(is.na(host_location), "Unknown", host_location),
    host_response_time = ifelse(is.na(host_response_time), "Unknown", host_response_time),
    market = ifelse(is.na(market), "Unknown", market),
    zipcode = ifelse(is.na(zipcode),"Unknown",zipcode),
    smart_location = ifelse(is.na(smart_location),"Unknown",smart_location),
    jurisdiction_names = ifelse(is.na(jurisdiction_names), "unknown", jurisdiction_names),
    neighborhood_group = ifelse(is.na(neighborhood_group), "unknown", neighborhood_group),
    neighborhood= ifelse(is.na(neighborhood), "unknown", neighborhood),
    host_neighbourhood= ifelse(is.na(host_neighbourhood), "unknown", host_neighbourhood),
    host_name=ifelse(is.na(host_name), "unknown", host_name),
    city = as.factor(ifelse(is.na(city), "Missing",
                            ifelse(city_freq[train_rate$city] > 2000, city, "others"))),
    price_per_bedroom = price/bedrooms,
    price_per_accomodates = price/accommodates,
    license=  ifelse(is.na(license), "No License", "Has License"),
    transit = ifelse(is.na(transit), 0, 1),
    space= ifelse(is.na(space), 0, 1),
    summary = ifelse(is.na(summary), 0, 1),
    host_since_numeric = ifelse(is.na(host_since_numeric), 0, host_since_numeric)
  )

cleaning_tokenizer <- function(v) {
  v %>%
    space_tokenizer(sep = ',') 
}

#tokenize
it_token <- itoken(train_rate$amenities, 
                   preprocessor = tolower, #preprocessing by converting to lowercase
                   tokenizer = cleaning_tokenizer, 
                   progressbar = FALSE)

verify_token <- itoken(train_rate$host_verifications, 
                       preprocessor = tolower, #preprocessing by converting to lowercase
                       tokenizer = cleaning_tokenizer, 
                       progressbar = FALSE)

feat_token <- itoken(train_rate$features, 
                     preprocessor = tolower, #preprocessing by converting to lowercase
                     tokenizer = cleaning_tokenizer, 
                     progressbar = FALSE)

#learn the vocabulary
vocab <- create_vocabulary(it_token)
verify_vocab <- create_vocabulary(verify_token)
feat_vocab <- create_vocabulary(feat_token)

#vectorize
vectorizer <- vocab_vectorizer(vocab)
verify_vectorizer <- vocab_vectorizer(verify_vocab)
feat_vectorizer <- vocab_vectorizer(feat_vocab)


dtm_amen <- create_dtm(it_token, vectorizer)
dim(dtm_amen)

dtm_verify <- create_dtm(verify_token, verify_vectorizer)
dim(dtm_verify)

dtm_feat <- create_dtm(feat_token, feat_vectorizer)
dim(dtm_feat)

# this is small enough to be represented as a regular dataframe
cluster_matrix <- data.frame(as.matrix(dtm_amen))
head(cluster_matrix)

verify_matrix <- data.frame(as.matrix(dtm_verify))
head(verify_matrix)

feat_matrix <- data.frame(as.matrix(dtm_feat))
head(feat_matrix)

train_rate <- cbind(train_rate, cluster_matrix, verify_matrix, feat_matrix)

test_x <- test_x %>%
  mutate(
    host_since_numeric = as.numeric(difftime(Sys.Date(), as.Date(host_since), units = "days")),
    host_total_listings_count = ifelse(is.na(host_total_listings_count), 
                                       median(host_total_listings_count, na.rm=TRUE), host_total_listings_count),
    host_total_listings_count = ifelse(host_total_listings_count == 0, 1, host_total_listings_count),
    availability_365 = ifelse(availability_365 == 0, NA, availability_365),
    availability_365 = ifelse(is.na(availability_365), median(availability_365, na.rm=TRUE), availability_365),
    availability_30 = ifelse(availability_30 == 0, NA, availability_30),
    availability_30 = ifelse(is.na(availability_30), median(availability_30, na.rm=TRUE), availability_30),
    availability_60 = ifelse(availability_60 == 0, NA, availability_60),
    availability_60 = ifelse(is.na(availability_60), median(availability_60, na.rm=TRUE), availability_60),
    availability_90 = ifelse(availability_90 == 0, NA, availability_90),
    availability_90 = ifelse(is.na(availability_90), median(availability_90, na.rm=TRUE), availability_90),
    
    bathrooms = ifelse(is.na(bathrooms), median(bathrooms, na.rm = TRUE), bathrooms),
    security_deposit = ifelse(is.na(security_deposit), median(security_deposit, na.rm = TRUE), security_deposit),
    host_acceptance_rate = parse_number(host_acceptance_rate),
    accommodates = ifelse(is.na(accommodates), mean(accommodates, na.rm=TRUE), accommodates),
    host_acceptance = as.factor(ifelse(is.na(host_acceptance_rate), "MISSING",
                                       ifelse(host_acceptance_rate == 100, "ALL", "SOME"))),
    host_response = as.factor(ifelse(is.na(host_response_rate), "MISSING",
                                     ifelse(host_response_rate == 100, "ALL",  "SOME"))),
    beds=ifelse(is.na(beds),mean(beds,na.rm=TRUE),beds),
    cleaning_fee=ifelse(is.na(cleaning_fee),0,cleaning_fee),
    bathrooms = ifelse(is.na(bathrooms), median(bathrooms, na.rm = TRUE), bathrooms),
    property_category = as.factor(case_when(
      property_type %in% c("Apartment", "Serviced apartment", "Loft") ~ "apartment",
      property_type %in% c("Bed & Breakfast", "Boutique hotel", "Hostel") ~ "hotel",
      property_type %in% c("Townhouse", "Condominium") ~ "condo",
      property_type %in% c("Bungalow", "House") ~ "house",
      TRUE ~ "other")),
    state = as.factor(case_when(
      state %in% c("ca", "Ca", "CA") ~ "CA",
      state %in% c("il", "IL") ~ "IL",
      state %in% c("ny", "NY") ~ "NY",
      state %in% c("CO") ~ "CO",
      state %in% c("LA") ~ "LA",
      state %in% c("MA") ~ "MA",
      state %in% c("MD") ~ "MD",
      state %in% c("OR") ~ "OR",
      state %in% c("TN") ~ "TN",
      state %in% c("TX") ~ "TX",
      state %in% c("WA") ~ "WA",
      TRUE ~ "MISSING"
    )),
    bed_type = as.factor(bed_type),
    bed_category = as.factor(ifelse(bed_type == "Real Bed", "Bed", "Other")),
    room_type = as.factor(room_type),
    maximum_nights=ifelse(is.na(maximum_nights), mean(maximum_nights, na.rm = TRUE), maximum_nights),
    cancellation_policy = as.factor(ifelse(cancellation_policy %in% c("super_strict_60", "super_strict_30"), 
                                           "strict", cancellation_policy)),
    price = ifelse(is.na(price), mean(price, na.rm = TRUE), price),
    latitude = ifelse(is.na(latitude), 0, latitude),
    longitude = ifelse(is.na(longitude), 0, longitude),
    country=ifelse(is.na(country), "United States", country),
    country_code= ifelse(is.na(country_code), "US", country_code),
    host_location = ifelse(is.na(host_location), "Unknown", host_location),
    host_response_time = ifelse(is.na(host_response_time), "Unknown", host_response_time),
    market = ifelse(is.na(market), "Unknown", market),
    zipcode = ifelse(is.na(zipcode),"Unknown",zipcode),
    smart_location = ifelse(is.na(smart_location),"Unknown",smart_location),
    jurisdiction_names = ifelse(is.na(jurisdiction_names), "unknown", jurisdiction_names),
    neighborhood_group = ifelse(is.na(neighborhood_group), "unknown", neighborhood_group),
    neighborhood= ifelse(is.na(neighborhood), "unknown", neighborhood),
    host_neighbourhood= ifelse(is.na(host_neighbourhood), "unknown", host_neighbourhood),
    host_name=ifelse(is.na(host_name), "unknown", host_name),
    monthly_price= ifelse(is.na(monthly_price), mean(monthly_price, na.rm = TRUE), monthly_price),
    square_feet=ifelse(is.na(square_feet), mean(square_feet, na.rm = TRUE), square_feet),
    weekly_price=ifelse(is.na(weekly_price), mean(weekly_price, na.rm = TRUE), weekly_price),
    guests_included=ifelse(is.na(guests_included), median(guests_included, na.rm = TRUE), guests_included),
    city = as.factor(ifelse(is.na(city), "Missing",
                            ifelse(city_freq[train_rate$city] > 2000, city, "others"))),
    price_per_bedroom = price/bedrooms,
    price_per_accomodates = price/accommodates,
    license=  ifelse(is.na(license), "No License", "Has License"),
    transit = ifelse(is.na(transit), 0, 1),
    space= ifelse(is.na(space), 0, 1),
    summary = ifelse(is.na(summary), 0, 1),
    host_since_numeric = ifelse(is.na(host_since_numeric), 0, host_since_numeric)
  )

cleaning_tokenizer_test <- function(v) {
  v %>%
    space_tokenizer(sep = ',') 
}

#tokenize
it_test <- itoken(test_x$amenities, 
                  preprocessor = tolower, #preprocessing by converting to lowercase
                  tokenizer = cleaning_tokenizer_test, 
                  progressbar = FALSE)

it_verify <- itoken(test_x$host_verifications, 
                    preprocessor = tolower, #preprocessing by converting to lowercase
                    tokenizer = cleaning_tokenizer_test, 
                    progressbar = FALSE)

it_feat <- itoken(test_x$features, 
                  preprocessor = tolower, #preprocessing by converting to lowercase
                  tokenizer = cleaning_tokenizer_test, 
                  progressbar = FALSE)


#vectorize
dtm_test <- create_dtm(it_test, vectorizer)
dim(dtm_test)

dtm_test_verify <- create_dtm(it_verify, verify_vectorizer)
dim(dtm_test_verify)

dtm_test_feat <- create_dtm(it_feat, feat_vectorizer)
dim(dtm_test_feat)

# this is small enough to be represented as a regular dataframe
cluster_matrix_test <- data.frame(as.matrix(dtm_test))
head(cluster_matrix_test)

verify_matrix_test <- data.frame(as.matrix(dtm_test_verify))
head(verify_matrix_test)

feat_matrix_test <- data.frame(as.matrix(dtm_test_feat))
head(feat_matrix_test)

test_x <- cbind(test_x, cluster_matrix_test, verify_matrix_test, feat_matrix_test)


# summary
summary(train_rate)
"host_response" %in% colnames(train_rate)


# Section 5: Splitting Data into Training and Validation Sets
valid_instn <- sample(nrow(train_rate), 0.10*nrow(train_rate))
test <- train_rate[valid_instn,]
rest <- train_rate[-valid_instn,]
split <- sample(nrow(rest), 0.30*nrow(rest))
train_final <- rest[-split,]
valid_final <- rest[split,]


dependent_var <- "high_booking_rate"
independent_vars <- c("bed_type", "cancellation_policy", "host_response", "property_category", 
                      "room_type", "accommodates", "availability_30", "availability_365", 
                      "availability_60", "availability_90", "bathrooms", "bedrooms", 
                      "cleaning_fee", "extra_people", "host_acceptance", "host_listings_count", 
                      "host_total_listings_count", "minimum_nights", "price",
                      "state", "latitude", "longitude", "city", "host_since_numeric")

cluster_vars <- colnames(cluster_matrix)
verify_vars <- colnames(verify_matrix)
feat_vars <- colnames(feat_matrix)
# Add cluster matrix variables to the independent_vars vector
independent_vars <- c(independent_vars, cluster_vars, verify_vars, feat_vars)


# Create the formula string with the updated independent_vars vector
formula_str <- paste(dependent_var, "~", paste(independent_vars, collapse = " + "))
formula <- as.formula(formula_str)


#MODEL 1 : LOGISTIC REGRESSION:

log_model <- glm(data = train_final, formula, family = "binomial")
summary(log_model)

log_prediction <- predict(log_model, newdata = valid_final, type = "response")
summary(log_prediction)

predicted_classes <- ifelse(log_prediction >= 0.5, 1, 0)

actual_classes <- ifelse(valid_final$high_booking_rate == "YES", 1, 0)

correct_classifications <- ifelse(predicted_classes == actual_classes, 1, 0)

log_accuracy <- sum(correct_classifications) / length(correct_classifications)
log_accuracy

lg <- prediction(log_prediction, valid_final$high_booking_rate)
performance(lg, measure = "auc")@y.values[[1]]

tp <- sum(predicted_classes[actual_classes == 1] == 1)
fp <- sum(predicted_classes[actual_classes == 0] == 1)
tn <- sum(predicted_classes[actual_classes == 0] == 0)
fn <- sum(predicted_classes[actual_classes == 1] == 0)

fpr <- fp / (fp + tn)
tpr <- tp / (tp + fn)

library(pROC)
roc <- roc(actual_classes, log_prediction)
plot.roc(roc, print.auc=TRUE)


#MODEL 2 TREE MODEL:
# Load libraries
library(rpart)

tree_models <- list()

cp_values <- seq(0.01, 0.1, by = 0.01)

for (cp in cp_values) {
  
  tree_model <- rpart(formula,
                      data = train_final,
                      method = "class",
                      control = rpart.control(cp = cp))
  
  
  tree_models[[as.character(cp)]] <- tree_model
}


best_cp <- NULL
best_accuracy <- 0

for (cp in cp_values) {
 
  predictions <- predict(tree_models[[as.character(cp)]], newdata = valid_final, type = "class")
  
 
  accuracy <- sum(predictions == valid_final$high_booking_rate) / nrow(valid_final)
  
 
  if (accuracy > best_accuracy) {
    best_accuracy <- accuracy
    best_cp <- cp
  }
}


best_tree_model <- tree_models[[as.character(best_cp)]]

cat("Best cp value:", best_cp, "\n")
cat("Best Accuracy:", best_accuracy, "\n")


tree_prediction <- predict(best_tree_model, newdata = valid_final, type = "prob")[, 2]

roc_tree <- roc(ifelse(valid_final$high_booking_rate == "YES", 1, 0), tree_prediction)
plot.roc(roc_tree, print.auc=TRUE)


cat("Tree Model AUC:", auc(roc_tree),"\n")

#Model 3: LASSO:


library(caret)
cleaned_train_x <- train_final%>%
  select(all_of(independent_vars))


encodings <- dummyVars(~ ., data = cleaned_train_x)
X <- predict(encodings, newdata = cleaned_train_x)
X <- as.data.frame(X)
X[] <- lapply(X, as.numeric)
X <- as.matrix(X)


cleaned_train_y <- train_final %>%
  select(high_booking_rate)


factor_vars_y <- names(cleaned_train_y)[sapply(cleaned_train_y, is.factor)]
print(factor_vars_y)
cleaned_train_y_numeric <- cleaned_train_y

for (var in factor_vars_y) {
  cleaned_train_y_numeric[[var]] <- as.numeric(cleaned_train_y_numeric[[var]])
}

cleaned_train_y_numeric <- as.matrix(cleaned_train_y_numeric)

lasso <- cv.glmnet(X, cleaned_train_y_numeric, alpha = 1, family="binomial")
plot(lasso, xvar = "norm", label = TRUE)

best_lambda <- lasso$lambda.min
print(best_lambda)

best_model <- glmnet(X, cleaned_train_y_numeric, family = "binomial", alpha = 1, lambda = best_lambda)
coefficients <- coef(best_model)
print(coefficients)

nonzero_indices <- which(coefficients != 0)
print(nonzero_indices)

zero_indices <- which(coefficients == 0)
print(zero_indices)


column_names <- rownames(coefficients)[nonzero_indices]
column_names <- column_names[column_names != "(Intercept)"]
print(column_names)


column_names_zerocoefficient <- rownames(coefficients)[zero_indices]
print(column_names_zerocoefficient)


validation_cleaned_data_x <- valid_final %>%
  select(all_of(independent_vars)
  )

validation_y <- valid_final %>%
  select(high_booking_rate)


encodings <- dummyVars(~ ., data = validation_cleaned_data_x)
X_val <- predict(encodings, newdata = validation_cleaned_data_x)
X_val <- as.data.frame(X_val)
X_val[] <- lapply(X_val, as.numeric)
X_val <- as.matrix(X_val)


pred <- predict(lasso, s=best_lambda, newx = X_val ,type="response")


lambdas <- lasso$lambda
errors <- lasso$cvm

plot(lambdas, errors)


plot(log(lambdas), errors)


classifications <- ifelse(pred > .5, "YES", "NO")


accuracy <- sum(classifications == validation_y) / length(classifications)
print(accuracy*100)

#MODEL 4 : RANDOM FOREST
rf.mod <- randomForest(formula,
                       data = train_final,
                       subset = NULL,
                       mtry = 6, ntree = 1000,
                       importance = TRUE)

rf_preds <- predict(rf.mod, newdata = valid_final)
rf_acc <- mean(ifelse(rf_preds == valid_final$high_booking_rate, 1, 0))

rf.mod
rf_acc

# Plot the variable importances
varImpPlot(rf.mod, scale = FALSE)


#MODEL 5 : BAGGING:

library(pROC)


rest$high_booking_rate <- as.numeric(rest$high_booking_rate) - 1

bag.mod <- randomForest(formula,
                        data = train_final,
                        subset = NULL,
                        mtry = 22, # Defaults to 500 trees
                        importance = TRUE) 


bag_preds <- predict(bag.mod, newdata = valid_final)
bag_acc <- mean(ifelse(bag_preds == valid_final$high_booking_rate, 1, 0))

bag.mod
bag_acc


#MODEL 6 : Boosting:

boost_data <- train_rate
boost_data$high_booking_rate <- as.numeric(boost_data$high_booking_rate) - 1
boost_train <- boost_data[-valid_instn,]
boost_valid <- boost_data[valid_instn,]

boost.mod <- gbm(formula,
                 data = boost_train,
                 distribution = "bernoulli",
                 n.trees = 1500,
                 interaction.depth = 10)
boost_preds <- predict(boost.mod, newdata = boost_valid, type = 'response', n.trees = 1000)

boost_class <- ifelse(boost_preds > .5, 1, 0)
boost_acc <- mean(ifelse(boost_class == boost_valid$high_booking_rate, 1, 0))
boost_acc

lg <- prediction(boost_preds, boost_valid$high_booking_rate)
roc_lg <- performance(lg, "tpr", "fpr")
plot(roc_lg, col="purple", lwd=2, main="ROC Curve", xlab="False Positive Rate", ylab="True Positive Rate")

performance(lg, measure = "auc")@y.values[[1]]


# Validation Curve for Boosting Model
library(gbm)
library(pROC)

calculate_auc <- function(predictions, actual_classes) {
  roc <- roc(actual_classes, predictions)
  return(auc(roc))
}

auc_values <- c()


n_trees_values <- seq(100, 1000, by = 100)


for (n_trees in n_trees_values) {
  
  boost.mod3 <- gbm(formula,
                   data = boost_train,
                   distribution = "bernoulli",
                   n.trees = n_trees,
                   interaction.depth = 10)
  
  
  valid_probs <- predict(boost.mod3, newdata = boost_valid, type = "response")
  
 
  valid_auc <- calculate_auc(valid_probs, boost_valid$high_booking_rate)
  auc_values <- c(auc_values, valid_auc)
}

plot(n_trees_values, auc_values, type = "l", col = "blue", xlab = "Number of Trees", ylab = "AUC", main = "Validation Curve")

# MODEL: Final model: BOOSTING 
boost_train2 <- train_rate
boost_train2$high_booking_rate <- as.numeric(boost_train2$high_booking_rate) - 1

boost.mod2 <- gbm(formula,
                  data = boost_train2,
                  distribution = "bernoulli",
                  n.trees = 1500,
                  interaction.depth = 10)


probs_rate <- predict(boost.mod2, newdata = test_x, type = "response")
probs_rate <- ifelse(is.na(probs_rate), mean(probs_rate), probs_rate)
assertthat::assert_that(sum(is.na(probs_rate))==0)
summary(probs_rate)


write.table(probs_rate, "high_booking_rate_group26.csv", row.names = FALSE)