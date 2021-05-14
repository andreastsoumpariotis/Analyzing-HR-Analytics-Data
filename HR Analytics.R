
### STAT 6302 Final Project ###

#Data: https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists


#Load packages
library(dplyr)
library(tidyr)
library(plyr)
library(ggplot2)
library(lme4)
library(AICcmodavg)
library(magrittr)
library(scales)
library(gmodels)
library(agricolae)
library(multcomp)
library(Sleuth2)
library(MASS)
library(car)
library(glmnet)
library(caret)
library(leaps)
library(bestglm)
library(VIM)
library("VIM")
library(forcats)
library(stringr)
library(formattable)
library(MASS)
library(AppliedPredictiveModeling)
library(klaR)
library(randomForest)
library(pROC)
library(OptimalCutpoints)
library(ISLR)
library(naniar)
library(factoextra)
library(lsmeans)
library(tidyverse)
library(Sleuth3)
library(tree)
library(ranger)
library(partykit)
library(parallel)
library(doParallel)
library(xgboost)
library(kernlab)
library(MLmetrics)
library(smotefamily)
library(DMwR)
library(rminer)


#Read in data (make sure to read in 'blanks' as NAs)

#Train data
aug = read.csv("aug_train.csv", header=T, na.strings=c(""," ","NA"))
aug = data.frame(aug)
head(aug)
str(aug) #19158 obs & 14 variables


#Inspect predictor and target variables

#Inspect 'city' variable
aug$city = as.factor(aug$city)
aug %>% count('city') #categorical w/123 unique levels (or cities)

#Inspect 'city_development_index' variable
range(aug$city_development_index) #values range from 0.448 to 0.949

#Inspect 'gender' variable
aug$gender = as.factor(aug$gender)
levels(aug$gender) #male, female, or other

#Inspect 'relevent_experience' variable
aug$relevent_experience = as.factor(aug$relevent_experience)
levels(aug$relevent_experience) #relevant work experience or not; 0 = no experience / 1 = experience

#Inspect 'enrolled_university' variable
aug$enrolled_university = as.factor(aug$enrolled_university)
levels(aug$enrolled_university) #full time, part time, or no enrollment

#Inspect 'education_level' variable
aug$education_level = as.factor(aug$education_level)
table(aug$education_level) #Graduate, High School, Masters, PhD, or Primary School

#Inspect 'major_discipline' variable
aug$major_discipline = as.factor(aug$major_discipline)
aug %>% count('major_discipline') #STEM, Business, Humanities, Arts, Other, or No Major

#Inspect 'experience' variable
aug$experience = as.factor(aug$experience)
table(aug$experience) #grouped between less than 1 and greater than 20

#Inspect 'company_size' variable
aug$company_size = as.factor(aug$company_size)
aug %>% count('company_size') #grouped between less than 10 and 5000-9999 employees

#Inspect 'company_type' variable
aug$company_type = as.factor(aug$company_type)
table(aug$company_type) #Early-Stage Startup, Funded Startup, NGO, Public Sector, Private Ltd, or Other

#Inspect 'last_new_job' variable
aug$last_new_job = as.factor(aug$last_new_job)
table(aug$last_new_job) #grouped between 1 and greater than 4 years, as well as never

#Inspect 'training_hours' variable
range(aug$training_hours) #values range from 1 to 336 hours

#Inspect 'target' variable (target variable)
aug$target = as.factor(aug$target)
levels(aug$target) #0 = Not looking for a job change, 1 = Looking for a job change (predicting '1')


#Drop 'enrollee_id' and 'city' variables
aug = aug[,-c(1,2)]
str(aug) #19158 obs & 12 variables


#Near-zero variance variables
near_zero = nearZeroVar(aug)
near_zero

#The 'major_discipline' variable has near-zero variance- remove it
aug = aug[,-6]
str(aug) #19158 obs & 11 variables


#Check to see if any class imbalances exist- there are some class imbalances but we'll proceed w/caution
table(aug$gender) #Male = 13221; Female = 1238; Other = 191
table(aug$relevent_experience) #Relevant experience = 13792; No relevant experience = 5366
table(aug$education_level) #Graduate = 11598; HS = 2017, Masters = 4361, Phd = 414, Primary School = 308
table(aug$enrolled_university) #Full time = 3757, no_enrollment = 13817, Part time = 1198

#Barplots to show variables w/class imbalances
par(mfrow=c(2,2))

#Gender
barplot(table(aug$gender),main="Barplot of Gender",
        xlab="Gender Type",
        ylab="Count",
        col="black")
#Relevant Experience
barplot(table(aug$relevent_experience),main="Barplot of Relevant Experience",
        xlab="Relevant Experience",
        ylab="Count",
        col="black")
#Education Level
barplot(table(aug$education_level),main="Barplot of Education Level",
        xlab="Education Level",
        ylab="Count",
        col="black")
#University Enrollment
barplot(table(aug$enrolled_university),main="Barplot of University Enrollment",
        xlab="Enrollment Type",
        ylab="Count",
        names.arg=c("Full time course"="Full time", "no_enrollment"="No enrollment", "Part time course"="Part time"),
        col="black")

#Change graph fit back
par(mfrow=c(1,1))


# Feature Engineering #

#See if variables have NAs
sapply(aug, function(x) sum(is.na(x))) #there 7 variables with NAs
#gender = 4508
#enrolled_university = 386
#education_level = 460
#experience = 65
#company_size = 5938
#company_type = 6140
#last_new_job = 423

#Change NAs to 'Other' for 'gender' variable
aug$gender[is.na(aug$gender)] <- "Other"
table(aug$gender)

#Change NAs to 'no_enrollment' for 'enrolled_university' variable
aug$enrolled_university[is.na(aug$enrolled_university)] <- "no_enrollment"
table(aug$enrolled_university)

#Use mode imputation for 'education_level', 'experience', 'company_size', 'company_type', & 
#'last_new_job' variables
c1<-makePSOCKcluster(3)
registerDoParallel(c1)
aug = VIM::kNN(aug, k = 10, numFun = mode)
aug = aug[,-c(12:22)]
str(aug)
sapply(aug, function(x) sum(is.na(x))) #no more NAs


#Change levels for certain variables

#Change 'education_level' levels
aug$education_level = ifelse(aug$education_level == "Graduate" | aug$education_level == "Masters" | aug$education_level == "Phd", 1, 0)
aug$education_level[aug$education_level == 1] <- "University"
aug$education_level[aug$education_level == 0] <- "Other"
aug$education_level = as.factor(aug$education_level)
levels(aug$education_level)
table(aug$education_level) 
#count: 1/uni = 16668, 0/other = 2490

#Change 'experience' levels
levels(aug$experience) <- list("Some Experience"=c("<1","1","2","3","4"), "Experience"=c("5","6","7","8","9","10","11","12","13","14"), "A lot of Experience"=c("15","16","17","18","19","20",">20"))
str(aug$experience)
table(aug$experience)
#count: Some Experience = 4983, Experience = 8604, A lot of Experience = 5571

#Change 'company_size' levels
levels(aug$company_size) <- list("Small"=c("<10","10/49","50-99","100-500"), "Medium"=c("500-999","1000-4999"), "Large"=c("5000-9999","10000+"))
str(aug$company_size)
table(aug$company_size)
#count: Small = 13036, Medium = 2675, Large = 3447

#Change 'company_type' levels
levels(aug$company_type) <- list("Public"=c("Public Sector"), "Startup"=c("Early Stage Startup","Funded Startup"), "Other"=c("Other","NGO","Pvt Ltd"))
str(aug$company_type)
table(aug$company_type)
#count: Public = 1157, Startup = 1681, Other = 16320

#Change 'last_new_job' levels
levels(aug$last_new_job) <- list("Never"=c("never"), "Recent"=c("1","2"), "Kind of Recent"=c("3","4"), "Not Recent"=c(">4"))
str(aug$last_new_job)
table(aug$last_new_job)
#count: Never = 2555, Recent = 11216, Kind of Recent = 2056, Not Recent = 3331


#Make 'city_development_index' out of 100 for better coefficient interpretation (logistic reg)
aug$city_development_index = 100*(aug$city_development_index)


# Train and Test Split #

#Inspect data before splitting
str(aug) #19158 obs & 11 variables

#Check to see which level we're predicting
levels(aug$target)
#reference = "0" (Not looking for job change), predicting = "1" (Looking for job change)

#Split data into 60% train, 25% validation, and 15% test
set.seed(100)
splitSample <- sample(1:3, size=nrow(aug), prob=c(0.6,0.25,0.15), replace = TRUE)
aug.train <- aug[splitSample==1,]
aug.valid <- aug[splitSample==2,]
aug.test <- aug[splitSample==3,]


# SMOTE #

#Inspect data before using SMOTE
str(aug.train) #11555 obs & 11 variables
str(aug.valid) #4800 obs & 11 variables
str(aug.test) #2803 obs & 11 variables

#Use SMOTE for train and test data
set.seed(100)
#Train
aug.train <- SMOTE(target ~ ., data = aug.train)                         
table(aug.train$target)
#Valid
aug.valid <- SMOTE(target ~ ., data = aug.valid)                         
table(aug.valid$target)
#Test
aug.test <- SMOTE(target ~ ., data = aug.test)                         
table(aug.test$target)

#Inspect data after using SMOTE
str(aug.train) #20440 obs & 11 variables
str(aug.valid) #8176 obs & 11 variables
str(aug.test) #4823 obs & 11 variables

#Compare class imbalances for 'target' post and pre SMOTE
table(aug.train$target)
table(aug$target)
par(mfrow=c(2,2))
#Post-SMOTE
barplot(table(aug.train$target),main="Barplot of Target (SMOTE)",
        xlab="Looking for Job Change",
        ylab="Count",
        names.arg=c("Not.Looking"="Not Looking", "Looking"="Looking"),
        col="black")
#Pre-SMOTE
barplot(table(aug$target),main="Barplot of Target",
        xlab="Looking for Job Change",
        ylab="Count",
        names.arg=c("0"="Not Looking", "1"="Looking"),
        col="black")
par(mfrow=c(1,1))


# Models #

#Logistic regression
c1<-makePSOCKcluster(3)
registerDoParallel(c1)
start_time <- Sys.time()

set.seed(100)
trControl = trainControl(method = "repeatedcv", repeats = 5)
logisticReg <- train(target ~ ., data = aug.train,
                     method = "glm", trControl = trControl)
summary(logisticReg)
logisticReg

end_time <- Sys.time()
end_time - start_time #13.28301 sec

#Create probabilities
aug.valid$prob <- predict(logisticReg, aug.valid, type = "prob")[, "1"]
range(aug.valid$prob)
aug.valid$pred <- rep("0", 8176)
aug.valid$pred[aug.valid$prob > 0.5] <- "1"
aug.valid$pred = as.factor(aug.valid$pred)

#Optimal cut point for maximizing Kappa
levels(aug.valid$target)
summary(optimal.cutpoints(X="prob", status="target", data=aug.valid, 
                          tag.healthy='0', methods='MaxKappa')) #optimal cut point = 0.3799812

#Predict on test set
logisticResults <- data.frame(obs = aug.test$target)
logisticResults$prob <- predict(logisticReg, aug.test, type = "prob")[, "1"]
range(logisticResults$prob)
logisticResults$pred <- rep("0", 4823)
logisticResults$pred[logisticResults$prob >  0.3799812] <- "1"
logisticResults$pred= as.factor(logisticResults$pred)
logisticResults

#Confusion Matrix for accuracy and kappa
confusionMatrix(data = logisticResults$pred, reference = logisticResults$obs, positive = "1")

#ROC Plot
augROC <- roc(relevel(logisticResults$obs, "0"), logisticResults$prob)
auc(augROC)
ci.auc(augROC)
plot(augROC, legacy.axes = TRUE, asp = NA)

#Brier Score
aug.brier = ifelse(logisticResults$obs == "1", 1, 0)
diff.squared = (logisticResults$prob-aug.brier)^2
length = length(logisticResults$prob)
brier = (1/length)*sum(diff.squared)
brier

#Accuracy=0.7155
#Kappa=0.4304
#AUC=0.7802
#Brier=0.1891764
#Sens=0.7470
#Spec=0.6919

#Calibration plot
calData.logistic <- calibration(obs ~ (1-prob), data = logisticResults, cuts = 10)
xyplot(calData.logistic, auto.key = list(columns = 2), main="Logistic Regression Calibration Plot")


#XGBoost (Stochastic Gradient Boosting)

#Change target variable for train, validation, and test data so model can "interpret" it

#Train
levels(aug.train$target) <- list("Looking"=c("1"), "Not.Looking"=c("0"))
aug.train <- within(aug.train, target <- relevel(target, ref = "Not.Looking"))
levels(aug.train$target)
#Validation
levels(aug.valid$target) <- list("Looking"=c("1"), "Not.Looking"=c("0"))
aug.valid <- within(aug.valid, target <- relevel(target, ref = "Not.Looking"))
levels(aug.valid$target)
#Test
levels(aug.test$target) <- list("Looking"=c("1"), "Not.Looking"=c("0"))
aug.test <- within(aug.test, target <- relevel(target, ref = "Not.Looking"))
levels(aug.test$target)

#Fit boosting model
c1<-makePSOCKcluster(3)
registerDoParallel(c1)
start_time <- Sys.time()

trnCtrl <- trainControl(method = "repeatedcv", classProbs = TRUE, number=10, repeats=5, 
                        savePredictions = TRUE)


set.seed(100)
my.train.xgb <- train(target ~ ., data = aug.train, 
                      method="xgbTree", trControl=trnCtrl, tuneLength=3)

end_time <- Sys.time()
end_time - start_time #25.35039 min

my.train.xgb$results

#Optimal tuning parameters
my.train.xgb$bestTune

#Create probabilities
aug.valid$prob2 <- predict(my.train.xgb, aug.valid[,1:11], type = "prob")[, "Looking"]
range(aug.valid$prob2)
aug.valid$pred2 <- rep("Not.Looking", 8176)
aug.valid$pred2[aug.valid$prob2 > 0.5] <- "Looking"
aug.valid$pred2 = as.factor(aug.valid$pred2)

#Optimal cut point for maximizing Kappa
levels(aug.valid$target)
summary(optimal.cutpoints(X="prob2", status="target", data=aug.valid, 
                          tag.healthy='Not.Looking', methods='MaxKappa')) #optimal cut point = 0.5522765

#Predict on test set
boostingResults <- data.frame(obs = aug.test$target)
boostingResults$prob <- predict(my.train.xgb, aug.test, type = "prob")[, "Looking"]
boostingResults$pred <- rep("Not.Looking", 4823)
boostingResults$pred[boostingResults$prob > 0.5522765] <- "Looking"
boostingResults$pred = as.factor(boostingResults$pred)
range(boostingResults$prob)

#Confusion Matrix for accuracy and kappa
xgbCM <- confusionMatrix(data=boostingResults$pred, reference=boostingResults$obs, positive = "Looking")
xgbCM

#ROC
xgbRoc <- roc(response = boostingResults$obs, predictor = boostingResults$prob)
xgbRoc
plot(xgbRoc, type = "s", main = "Boosting ROC Plot", col = rgb(.2, .2, .2, .2), legacy.axes = TRUE, asp = NA)
plot(xgbRoc, type = "s", add = TRUE, print.thres = c(.5), 
     print.thres.pch = 3, legacy.axes = TRUE, print.thres.pattern = "", 
     print.thres.cex = 1.2,
     col = "red", print.thres.col = "red")

#Brier Score
aug.brier2 = ifelse(boostingResults$obs == "Looking", 1, 0)
diff.squared2 = (boostingResults$prob-aug.brier2)^2
length2 = length(boostingResults$prob)
brier2 = (1/length2)*sum(diff.squared2)
brier2

#Accuracy=0.819
#Kappa=0.6236
#AUC=0.8936
#Brier=0.1254407
#Sens=0.7160
#Spec=0.8962

#Calibration plot
calData.boosting <- calibration(obs ~ (1-prob), data = boostingResults, cuts = 10)
xyplot(calData.boosting, auto.key = list(columns = 2), main="Boosting Calibration Plot")

#Variable Importance
boosting.imp <- xgb.importance(feature_names=aug.train$feature_names,model=my.train.xgb$finalModel)
xgb.plot.importance(boosting.imp, main="Variable Importance Plot", xlab="Importance") #plot
boosting.imp #variable importance table


#Penalized logistic regression
c1<-makePSOCKcluster(3)
registerDoParallel(c1)
start_time <- Sys.time()

set.seed(100)
glmnGrid<-expand.grid(alpha=c(0, 0.5, 1),
                      lambda=seq(0.01, 0.2, length=10))
ctrl<-trainControl(method="repeatedcv", classProbs=TRUE, repeats=5,
                   summaryFunction=twoClassSummary, savePredictions=TRUE)
glmnFit<-train(target ~ .,
               data = aug.train,
               method="glmnet",
               trControl=ctrl,
               tuneGrid=glmnGrid,
               metric="ROC",
               preProc=c("center", "scale"),
               family="binomial")

end_time <- Sys.time()
end_time - start_time #57.74767 sec

glmnFit$results
coef(glmnFit$finalModel, s=glmnFit$bestTune$lambda)

#Optimal tuning parameters
glmnFit$bestTune #Ridge is the best model (alpha=0, lambda=0.01)

#Create probabilities
aug.valid$prob3 <- predict(glmnFit, aug.valid[,1:11], type = "prob")[, "Looking"]
range(aug.valid$prob3)
aug.valid$pred3 <- rep("Not.Looking", 8176)
aug.valid$pred3[aug.valid$prob3 > 0.5] <- "Looking"
aug.valid$pred3 = as.factor(aug.valid$pred3)

#Optimal cut point for maximizing Kappa
levels(aug.valid$target)
summary(optimal.cutpoints(X="prob3", status="target", data=aug.valid, 
                          tag.healthy='Not.Looking', methods='MaxKappa'))
#optimal cut point = 0.3493286

#Predict on test set
glmResults <- data.frame(obs = aug.test$target)
glmResults$prob <- predict(glmnFit, aug.test, type = "prob")[, "Looking"]
glmResults$pred <- rep("Not.Looking", 4823)
glmResults$pred[glmResults$prob > 0.3493286] <- "Looking"
glmResults$pred = as.factor(glmResults$pred)
range(glmResults$prob)

#Confusion Matrix for accuracy and kappa
confusionMatrix(data = glmResults$pred, reference = glmResults$obs, positive = "Looking")

#ROC Plot
glmROC <- roc(relevel(glmResults$obs, "Not.Looking"), glmResults$prob)
auc(glmROC)
ci.auc(glmROC)
plot(glmROC, legacy.axes = TRUE, asp = NA)

#Brier Score
glm.brier = ifelse(glmResults$obs == "Looking", 1, 0)
diff.squared3 = (glmResults$prob-glm.brier)^2
length3 = length(glmResults$prob)
brier3 = (1/length3)*sum(diff.squared3)
brier3

#Accuracy=0.7159
#Kappa=0.4383
#AUC=0.7804
#Brier=0.1893374
#Sens=0.7983
#Spec=0.6542

#Calibration plot
calData.glm <- calibration(obs ~ (1-prob), data = glmResults, cuts = 10)
xyplot(calData.glm, auto.key = list(columns = 2), main="Penalized Logistic Regression Calibration Plot")


#Random Forest
rf_grid <- expand.grid(mtry = c(2, 4, 6, 8, 10),
                       splitrule = c("variance", "extratrees"),
                       min.node.size = c(1, 3, 5))

c1<-makePSOCKcluster(3)
registerDoParallel(c1)

start_time <- Sys.time()

set.seed(100)
ctrl <- trainControl(method = "repeatedcv", classProbs = TRUE, number=10, repeats=5, savePredictions = TRUE)
rfFit <- train(target ~ .,
               data = aug.train,
               method = "ranger",
               tuneGrid = rf_grid,
               importance = 'impurity',
               metric = "Kappa",
               trControl = ctrl)

end_time <- Sys.time()
end_time - start_time #1.509027 hours

#Create probabilities
aug.valid$prob4 <- predict(rfFit, aug.valid[,1:11], type = "prob")[, "Looking"]
range(aug.valid$prob4)
aug.valid$pred4 <- rep("Not.Looking", 8176)
aug.valid$pred4[aug.valid$prob4 > 0.5] <- "Looking"
aug.valid$pred4 = as.factor(aug.valid$pred4)

#Optimal cut point for maximizing Kappa
levels(aug.valid$target)
summary(optimal.cutpoints(X="prob4", status="target", data=aug.valid, 
                          tag.healthy='Not.Looking', methods='MaxKappa'))
#optimal cut point = 0.4430048

#Predict on test set
rfResults <- data.frame(obs = aug.test$target)
rfResults$prob <- predict(rfFit, aug.test, type = "prob")[, "Looking"]
rfResults$pred <- rep("Not.Looking", 4823)
rfResults$pred[rfResults$prob > 0.4430048] <- "Looking"
rfResults$pred = as.factor(rfResults$pred)
range(rfResults$prob)

#Confusion Matrix for accuracy and kappa
confusionMatrix(data = rfResults$pred, reference = rfResults$obs, positive = "Looking")

#ROC Plot
rfROC <- roc(relevel(rfResults$obs, "Not.Looking"), rfResults$prob)
auc(rfROC)
ci.auc(rfROC)
plot(rfROC, legacy.axes = TRUE, asp = NA)

#Brier Score
rf.brier = ifelse(rfResults$obs == "Looking", 1, 0)
diff.squared4 = (rfResults$prob-rf.brier)^2
length4 = length(rfResults$prob)
brier4 = (1/length4)*sum(diff.squared4)
brier4

#Accuracy=0.7356
#Kappa=0.4692
#AUC=0.8027
#Brier=0.1812374
#Sens=0.7586
#Spec=0.7184

#Calibration plot
calData.rf <- calibration(obs ~ (1-prob), data = rfResults, cuts = 10)
xyplot(calData.rf, auto.key = list(columns = 2), main="Random Forest Calibration Plot")

#Calculate and plot variable importance (1st create 2nd rf model to use 'importance' function)
rf2 = randomForest(target ~ .,data=aug.train,importance=TRUE)
randomForest::importance(rf2)
varImpPlot(rf2, main="RF Variable Importance Plot")


#Support vector machine

#Get rid of all "prob" and "pred" variables
aug.valid = aug.valid[,1:11]
str(aug.valid)


#Change predictors to numeric so model can interpret it

#Train data

#gender (3=male, 2=female, 1=other)
levels(aug.train$gender) <- list("3"=c("Male"), "2"=c("Female"), "1"=c("Other"))
levels(aug.train$gender)
aug.train$gender = as.numeric(as.character(aug.train$gender))
str(aug.train$gender)
#relevent_experience (2=Has relevent experience, 1=No relevent experience)
levels(aug.train$relevent_experience) <- list("2"=c("Has relevent experience"), "1"=c("No relevent experience"))
levels(aug.train$relevent_experience)
aug.train$relevent_experience = as.numeric(as.character(aug.train$relevent_experience))
str(aug.train$relevent_experience)
#enrolled_university (3=Full time, 2=Part time, 1=no_enrollment)
levels(aug.train$enrolled_university) <- list("3"=c("Full time course"), "2"=c("Part time course"), "1"=c("no_enrollment"))
levels(aug.train$enrolled_university)
aug.train$enrolled_university = as.numeric(as.character(aug.train$enrolled_university))
str(aug.train$enrolled_university)
#education_level (2=University, 1=Other)
levels(aug.train$education_level) <- list("2"=c("University"), "1"=c("Other"))
levels(aug.train$education_level)
aug.train$education_level = as.numeric(as.character(aug.train$education_level))
str(aug.train$education_level)
#experience (3=A lot of experience, 2=Experience, 1=Some Experience)
levels(aug.train$experience) <- list("3"=c("A lot of Experience"), "2"=c("Experience"), "1"=c("Some Experience"))
levels(aug.train$experience)
aug.train$experience = as.numeric(as.character(aug.train$experience))
str(aug.train$experience)
#company_size (3=Large, 2=Medium, 1=Small)
levels(aug.train$company_size) <- list("3"=c("Large"), "2"=c("Medium"), "1"=c("Small"))
levels(aug.train$company_size)
aug.train$company_size = as.numeric(as.character(aug.train$company_size))
str(aug.train$company_size)
#company_type (3=Public, 2=Startup, 1=Other)
levels(aug.train$company_type) <- list("3"=c("Public"), "2"=c("Startup"), "1"=c("Other"))
levels(aug.train$company_type)
aug.train$company_type = as.numeric(as.character(aug.train$company_type))
str(aug.train$company_type)
#last_new_job (4=Never, 3=Recent, 2=Kind of Recent, 1=Not Recent)
levels(aug.train$last_new_job) <- list("4"=c("Never"), "3"=c("Recent"), "2"=c("Kind of Recent"), "1"=c("Not Recent"))
levels(aug.train$last_new_job)
aug.train$last_new_job = as.numeric(as.character(aug.train$last_new_job))
str(aug.train$last_new_job)

#Validation data

#gender (3=male, 2=female, 1=other)
levels(aug.valid$gender) <- list("3"=c("Male"), "2"=c("Female"), "1"=c("Other"))
levels(aug.valid$gender)
aug.valid$gender = as.numeric(as.character(aug.valid$gender))
str(aug.valid$gender)
#relevent_experience (2=Has relevent experience, 1=No relevent experience)
levels(aug.valid$relevent_experience) <- list("2"=c("Has relevent experience"), "1"=c("No relevent experience"))
levels(aug.valid$relevent_experience)
aug.valid$relevent_experience = as.numeric(as.character(aug.valid$relevent_experience))
str(aug.valid$relevent_experience)
#enrolled_university (3=Full time, 2=Part time, 1=no_enrollment)
levels(aug.valid$enrolled_university) <- list("3"=c("Full time course"), "2"=c("Part time course"), "1"=c("no_enrollment"))
levels(aug.valid$enrolled_university)
aug.valid$enrolled_university = as.numeric(as.character(aug.valid$enrolled_university))
str(aug.valid$enrolled_university)
#education_level (2=University, 1=Other)
levels(aug.valid$education_level) <- list("2"=c("University"), "1"=c("Other"))
levels(aug.valid$education_level)
aug.valid$education_level = as.numeric(as.character(aug.valid$education_level))
str(aug.valid$education_level)
#experience (3=A lot of experience, 2=Experience, 1=Some Experience)
levels(aug.valid$experience) <- list("3"=c("A lot of Experience"), "2"=c("Experience"), "1"=c("Some Experience"))
levels(aug.valid$experience)
aug.valid$experience = as.numeric(as.character(aug.valid$experience))
str(aug.valid$experience)
#company_size (3=Large, 2=Medium, 1=Small)
levels(aug.valid$company_size) <- list("3"=c("Large"), "2"=c("Medium"), "1"=c("Small"))
levels(aug.valid$company_size)
aug.valid$company_size = as.numeric(as.character(aug.valid$company_size))
str(aug.valid$company_size)
#company_type (3=Public, 2=Startup, 1=Other)
levels(aug.valid$company_type) <- list("3"=c("Public"), "2"=c("Startup"), "1"=c("Other"))
levels(aug.valid$company_type)
aug.valid$company_type = as.numeric(as.character(aug.valid$company_type))
str(aug.valid$company_type)
#last_new_job (4=Never, 3=Recent, 2=Kind of Recent, 1=Not Recent)
levels(aug.valid$last_new_job) <- list("4"=c("Never"), "3"=c("Recent"), "2"=c("Kind of Recent"), "1"=c("Not Recent"))
levels(aug.valid$last_new_job)
aug.valid$last_new_job = as.numeric(as.character(aug.valid$last_new_job))
str(aug.valid$last_new_job)

#Test data

#gender (3=male, 2=female, 1=other)
levels(aug.test$gender) <- list("3"=c("Male"), "2"=c("Female"), "1"=c("Other"))
levels(aug.test$gender)
aug.test$gender = as.numeric(as.character(aug.test$gender))
str(aug.test$gender)
#relevent_experience (2=Has relevent experience, 1=No relevent experience)
levels(aug.test$relevent_experience) <- list("2"=c("Has relevent experience"), "1"=c("No relevent experience"))
levels(aug.test$relevent_experience)
aug.test$relevent_experience = as.numeric(as.character(aug.test$relevent_experience))
str(aug.test$relevent_experience)
#enrolled_university (3=Full time, 2=Part time, 1=no_enrollment)
levels(aug.test$enrolled_university) <- list("3"=c("Full time course"), "2"=c("Part time course"), "1"=c("no_enrollment"))
levels(aug.test$enrolled_university)
aug.test$enrolled_university = as.numeric(as.character(aug.test$enrolled_university))
str(aug.test$enrolled_university)
#education_level (2=University, 1=Other)
levels(aug.test$education_level) <- list("2"=c("University"), "1"=c("Other"))
levels(aug.test$education_level)
aug.test$education_level = as.numeric(as.character(aug.test$education_level))
str(aug.test$education_level)
#experience (3=A lot of experience, 2=Experience, 1=Some Experience)
levels(aug.test$experience) <- list("3"=c("A lot of Experience"), "2"=c("Experience"), "1"=c("Some Experience"))
levels(aug.test$experience)
aug.test$experience = as.numeric(as.character(aug.test$experience))
str(aug.test$experience)
#company_size (3=Large, 2=Medium, 1=Small)
levels(aug.test$company_size) <- list("3"=c("Large"), "2"=c("Medium"), "1"=c("Small"))
levels(aug.test$company_size)
aug.test$company_size = as.numeric(as.character(aug.test$company_size))
str(aug.test$company_size)
#company_type (3=Public, 2=Startup, 1=Other)
levels(aug.test$company_type) <- list("3"=c("Public"), "2"=c("Startup"), "1"=c("Other"))
levels(aug.test$company_type)
aug.test$company_type = as.numeric(as.character(aug.test$company_type))
str(aug.test$company_type)
#last_new_job (4=Never, 3=Recent, 2=Kind of Recent, 1=Not Recent)
levels(aug.test$last_new_job) <- list("4"=c("Never"), "3"=c("Recent"), "2"=c("Kind of Recent"), "1"=c("Not Recent"))
levels(aug.test$last_new_job)
aug.test$last_new_job = as.numeric(as.character(aug.test$last_new_job))
str(aug.test$last_new_job)

#Examine data again
str(aug.train)
str(aug.valid)
str(aug.test)


#Radial kernel
ctrl <- trainControl(method = "repeatedcv",
                     summaryFunction = multiClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

#The sigest() function suggests values of sigma to use
#For a radial basis function
sigmaRangeFull <- sigest(as.matrix(aug.train[,1:10]))
svmRGridFull <- expand.grid(sigma =  as.vector(sigmaRangeFull),
                            C = 2^(-3:3))
c1<-makePSOCKcluster(3)
registerDoParallel(c1)

start_time <- Sys.time()

set.seed(100)
svmRFitFull <- train(target ~ .,
                     data = aug.train,
                     method = "svmRadial",
                     metric = "Kappa",
                     preProc = c("center", "scale"),
                     tuneGrid = svmRGridFull,
                     trControl = ctrl)

end_time <- Sys.time()
end_time - start_time #58.11819 min

#Optimal tuning parameters
svmRFitFull$bestTune

#Create probabilities
aug.valid$prob5 <- predict(svmRFitFull, aug.valid[,1:11], type = "prob")[, "Looking"]
range(aug.valid$prob5)
aug.valid$pred5 <- rep("Not.Looking", 8176)
aug.valid$pred5[aug.valid$prob5 > 0.5] <- "Looking"
aug.valid$pred5 = as.factor(aug.valid$pred5)

#Optimal cut point for maximizing Kappa
levels(aug.valid$target)
summary(optimal.cutpoints(X="prob5", status="target", data=aug.valid, 
                          tag.healthy='Not.Looking', methods='MaxKappa'))
#optimal cut point = 0.4236699

#Predict on test
svmResults <- data.frame(obs = aug.test$target)
svmResults$prob <- predict(svmRFitFull, aug.test, type = "prob")[, "Looking"]
svmResults$pred <- rep("Not.Looking", 4823)
svmResults$pred[svmResults$prob > 0.4236699] <- "Looking"
svmResults$pred = as.factor(svmResults$pred)
range(svmResults$prob)

#Confusion Matrix for accuracy and kappas
svmRCM <- confusionMatrix(data=svmResults$pred, reference=svmResults$obs, positive = "Looking")
svmRCM

#ROC
svmRoc <- roc(response = svmResults$obs, predictor = svmResults$prob)
svmRoc
plot(svmRoc, type = "s", col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(svmRoc, type = "s", add = TRUE, print.thres = c(.5), 
     print.thres.pch = 3, legacy.axes = TRUE, print.thres.pattern = "", 
     print.thres.cex = 1.2,
     col = "red", print.thres.col = "red")

#Brier Score
nba.brier5 = ifelse(svmResults$obs == "Looking", 1, 0)
diff.squared5 = (svmResults$prob-nba.brier5)^2
length5 = length(svmResults$prob)
brier5 = (1/length5)*sum(diff.squared5)
brier5

#Accuracy=0.7464
#Kappa=0.4788
#AUC=0.7927
#Brier=0.1802205
#Sens=0.6773
#Spec=0.7983

#Calibration plot
calData.svm <- calibration(obs ~ (1-prob), data = svmResults, cuts = 10)
xyplot(calData.svm, auto.key = list(columns = 2), main="SVM Calibration Plot")


