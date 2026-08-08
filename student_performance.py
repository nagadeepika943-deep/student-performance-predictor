import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("student_data.csv")


# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

X = data[[
    "study_hours",
    "attendance",
    "previous_score",
    "assignments_completed"
]]

y = data["final_score"]


# ==========================================
# 3. SPLIT DATA INTO TRAINING AND TESTING
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 4. CREATE THE MACHINE LEARNING MODEL
# ==========================================

model = LinearRegression()


# ==========================================
# 5. TRAIN THE MODEL
# ==========================================

model.fit(X_train, y_train)

print("Model trained successfully!")


# ==========================================
# 6. MAKE PREDICTIONS ON TEST DATA
# ==========================================

predictions = model.predict(X_test)

print("\nActual scores:")
print(y_test.values)

print("\nPredicted scores:")
print(predictions)


# ==========================================
# 7. EVALUATE THE MODEL
# ==========================================

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance:")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)


# ==========================================
# 8. PREDICT SCORE FOR A NEW STUDENT
# ==========================================

print("\n--- New Student Prediction ---")

study_hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance percentage: "))
previous_score = float(input("Enter previous score: "))
assignments_completed = float(
    input("Enter number of assignments completed: ")
)


# Create DataFrame for the new student

new_student = pd.DataFrame({
    "study_hours": [study_hours],
    "attendance": [attendance],
    "previous_score": [previous_score],
    "assignments_completed": [assignments_completed]
})


# Predict the score

predicted_score = model.predict(new_student)

print("\nPredicted Final Score:", predicted_score[0])


# ==========================================
# 9. VISUALIZATION
# ==========================================

# ==========================================
# 9. VISUALIZATION
# ==========================================

plt.scatter(
    data["study_hours"],
    data["final_score"],
    label="Students"
)

# Create predictions for all students
all_predictions = model.predict(X)

# Plot the regression line
plt.plot(
    data["study_hours"],
    all_predictions,
    label="Regression Line"
)

plt.xlabel("Study Hours")
plt.ylabel("Final Score")

plt.title("Study Hours vs Final Score")

plt.legend()

plt.show()

# ==========================================
# 10. ATTENDANCE VS FINAL SCORE
# ==========================================

plt.scatter(
    data["attendance"],
    data["final_score"]
)

plt.xlabel("Attendance (%)")
plt.ylabel("Final Score")

plt.title("Attendance vs Final Score")

plt.show()