import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Student Success AI",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Success AI")
st.subheader("Smart Campus Analytics: Predict, Optimize & Improve Student Success")
st.write(
    "An analytics platform that helps faculty identify students "
    "who may need academic or placement support."
)

st.info(
    "Demo project: scores are rule-based indicators to support "
    "faculty review, not definitive predictions."
)

# --------------------------------------------------
# 1. SAMPLE STUDENT DATA
# --------------------------------------------------

def generate_data():
    rng = np.random.default_rng(42)

    names = [
        "Aarav Sharma", "Ananya Reddy", "Rahul Kumar",
        "Priya Singh", "Sai Kiran", "Meera Patel",
        "Arjun Rao", "Sneha Das", "Vikram Naidu",
        "Ishita Roy", "Karthik Varma", "Diya Nair",
        "Rohan Gupta", "Sanjana Iyer", "Manoj Reddy",
        "Aisha Khan", "Teja Krishna", "Nithya Rao",
        "Aditya Jain", "Pooja Verma", "Harsha Babu",
        "Lavanya Das", "Nikhil Shah", "Keerthi Rani",
        "Sandeep Kumar", "Madhavi Reddy", "Abhinav Bose",
        "Divya Menon", "Pranav Joshi", "Swathi Rao"
    ]

    departments = ["CSE", "IT", "ECE", "EEE", "Mechanical"]
    rows = []

    for i, name in enumerate(names):
        rows.append({
            "student_id": f"STU{i+1:03d}",
            "name": name,
            "department": departments[i % len(departments)],
            "year": int(rng.integers(1, 5)),
            "cgpa": round(float(rng.uniform(4, 10)), 2),
            "attendance": round(float(rng.uniform(35, 100)), 1),
            "assignment_completion": round(float(rng.uniform(20, 100)), 1),
            "lms_logins_week": int(rng.integers(0, 12)),
            "events_participated": int(rng.integers(0, 7)),
            "aptitude_score": round(float(rng.uniform(20, 100)), 1),
            "coding_score": round(float(rng.uniform(15, 100)), 1),
            "mock_interview_score": round(float(rng.uniform(20, 100)), 1),
            "technical_skills": round(float(rng.uniform(25, 100)), 1),
            "soft_skills": round(float(rng.uniform(25, 100)), 1),
            "student_feedback": round(float(rng.uniform(1, 5)), 1)
        })

    # Add example students to demonstrate different scenarios.
    rows[2].update({
        "cgpa": 4.8,
        "attendance": 42,
        "assignment_completion": 35,
        "lms_logins_week": 1,
        "aptitude_score": 38,
        "coding_score": 30,
        "mock_interview_score": 32
    })

    rows[8].update({
        "cgpa": 8.8,
        "attendance": 91,
        "assignment_completion": 95,
        "aptitude_score": 42,
        "coding_score": 35,
        "mock_interview_score": 38
    })

    return pd.DataFrame(rows)


# --------------------------------------------------
# 2. CLEAN AND VALIDATE DATA
# --------------------------------------------------

REQUIRED_COLUMNS = [
    "student_id", "name", "department", "year", "cgpa",
    "attendance", "assignment_completion", "lms_logins_week",
    "events_participated", "aptitude_score", "coding_score",
    "mock_interview_score", "technical_skills", "soft_skills",
    "student_feedback"
]


def clean_data(df):
    df = df.copy()

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            if col == "student_id":
                df[col] = [f"STU{i+1:03d}" for i in range(len(df))]
            elif col == "name":
                df[col] = [f"Student {i+1}" for i in range(len(df))]
            elif col == "department":
                df[col] = "Not specified"
            elif col == "year":
                df[col] = 1
            elif col == "student_feedback":
                df[col] = 3
            else:
                df[col] = 0

    numeric_cols = [
        c for c in REQUIRED_COLUMNS
        if c not in ["student_id", "name", "department"]
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        median = df[col].median()
        df[col] = df[col].fillna(
            median if pd.notna(median) else 0
        )

    limits = {
        "cgpa": (0, 10),
        "attendance": (0, 100),
        "assignment_completion": (0, 100),
        "lms_logins_week": (0, 100),
        "events_participated": (0, 100),
        "aptitude_score": (0, 100),
        "coding_score": (0, 100),
        "mock_interview_score": (0, 100),
        "technical_skills": (0, 100),
        "soft_skills": (0, 100),
        "student_feedback": (1, 5),
        "year": (1, 8)
    }

    for col, (low, high) in limits.items():
        df[col] = df[col].clip(low, high)

    for col in ["student_id", "name", "department"]:
        df[col] = df[col].astype(str)

    return df


# --------------------------------------------------
# 3. CALCULATE STUDENT SUCCESS SCORE
# --------------------------------------------------

def calculate_success(df):
    df = df.copy()

    academic = df["cgpa"] * 10

    attendance = df["attendance"]

    lms = (
        df["assignment_completion"] * 0.7
        + np.clip(df["lms_logins_week"] / 10, 0, 1) * 30
    )

    engagement = (
        np.clip(df["events_participated"] / 5, 0, 1) * 100
    )

    placement = (
        df["aptitude_score"]
        + df["coding_score"]
        + df["mock_interview_score"]
    ) / 3

    skills = (
        df["technical_skills"] + df["soft_skills"]
    ) / 2

    feedback = (
        (df["student_feedback"] - 1) / 4
    ) * 100

    df["success_score"] = (
        academic * 0.30
        + attendance * 0.20
        + lms * 0.15
        + engagement * 0.10
        + placement * 0.15
        + skills * 0.07
        + feedback * 0.03
    ).round(1)

    return df


# --------------------------------------------------
# 4. IDENTIFY AT-RISK STUDENTS
# --------------------------------------------------

def identify_risk(df):
    df = df.copy()

    df["academic_flag"] = (
        (df["cgpa"] < 6)
        | (df["assignment_completion"] < 50)
    )

    df["attendance_flag"] = df["attendance"] < 65

    df["placement_flag"] = (
        (
            df["aptitude_score"]
            + df["coding_score"]
            + df["mock_interview_score"]
        ) / 3
    ) < 45

    df["engagement_flag"] = (
        (df["lms_logins_week"] < 2)
        | (df["events_participated"] == 0)
    )

    df["risk_factors"] = df.apply(
        lambda r: ", ".join([
            label
            for label, flag in [
                ("Low CGPA or assignments", r["academic_flag"]),
                ("Low attendance", r["attendance_flag"]),
                ("Low placement readiness", r["placement_flag"]),
                ("Low engagement", r["engagement_flag"])
            ]
            if flag
        ]) or "No major risk flags",
        axis=1
    )

    high_risk = (
        (
            df["academic_flag"]
            & df["attendance_flag"]
        )
        | (df["success_score"] < 45)
    )

    medium_risk = (
        (df["success_score"] < 60)
        | df["academic_flag"]
        | df["attendance_flag"]
        | df["placement_flag"]
    )

    df["risk_level"] = np.select(
        [high_risk, medium_risk],
        ["High", "Medium"],
        default="Low"
    )

    return df


# --------------------------------------------------
# 5. STUDENT SEGMENTATION
# --------------------------------------------------

def create_segment(row):
    placement = (
        row["aptitude_score"]
        + row["coding_score"]
        + row["mock_interview_score"]
    ) / 3

    if row["cgpa"] >= 7.5 and placement < 50:
        return "Strong academics, low placement readiness"

    if row["cgpa"] < 6 and row["attendance"] < 65:
        return "Academic and attendance support"

    if row["attendance"] >= 80 and row["cgpa"] >= 7.5:
        return "Consistent performer"

    if (
        row["events_participated"] == 0
        and row["lms_logins_week"] < 2
    ):
        return "Low engagement"

    return "Needs routine monitoring"


def suggest_action(row):
    actions = []

    if row["cgpa"] < 6 or row["assignment_completion"] < 50:
        actions.append("Faculty check-in and remedial study plan")

    if row["attendance"] < 65:
        actions.append("Discuss attendance barriers and follow-up")

    placement = (
        row["aptitude_score"]
        + row["coding_score"]
        + row["mock_interview_score"]
    ) / 3

    if placement < 50:
        actions.append("Aptitude practice, coding exercises and mock interviews")

    if row["lms_logins_week"] < 2 or row["events_participated"] == 0:
        actions.append("Check LMS access and encourage learning activities")

    if not actions:
        return "Monthly progress review and enrichment opportunities"

    return "; ".join(actions)


# --------------------------------------------------
# 6. LOAD DATA
# --------------------------------------------------

st.sidebar.header("Data Settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload student data CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        raw_data = pd.read_csv(uploaded_file)
        data_source = "Uploaded CSV"
    except Exception as error:
        st.sidebar.error(f"Unable to read CSV: {error}")
        raw_data = generate_data()
        data_source = "Sample data"
else:
    raw_data = generate_data()
    data_source = "Demo sample data"

data = clean_data(raw_data)
data = calculate_success(data)
data = identify_risk(data)

data["segment"] = data.apply(create_segment, axis=1)
data["suggested_action"] = data.apply(suggest_action, axis=1)

st.caption(
    f"Data source: {data_source} | "
    f"Total records: {len(data)}"
)


# --------------------------------------------------
# 7. FILTERS
# --------------------------------------------------

st.sidebar.subheader("Filters")

departments = sorted(data["department"].unique())

selected_departments = st.sidebar.multiselect(
    "Department",
    departments,
    default=departments
)

years = sorted(data["year"].unique())

selected_years = st.sidebar.multiselect(
    "Academic year",
    years,
    default=years
)

selected_risks = st.sidebar.multiselect(
    "Risk level",
    ["High", "Medium", "Low"],
    default=["High", "Medium", "Low"]
)

filtered = data[
    data["department"].isin(selected_departments)
    & data["year"].isin(selected_years)
    & data["risk_level"].isin(selected_risks)
].copy()

if filtered.empty:
    st.warning("No students match the selected filters.")
    st.stop()


# --------------------------------------------------
# 8. DASHBOARD METRICS
# --------------------------------------------------

st.header("📊 Student Performance Dashboard")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Students", len(filtered))

col2.metric(
    "Average Success Score",
    f"{filtered['success_score'].mean():.1f}/100"
)

col3.metric(
    "High-Risk Students",
    int((filtered["risk_level"] == "High").sum())
)

col4.metric(
    "Average Attendance",
    f"{filtered['attendance'].mean():.1f}%"
)


# --------------------------------------------------
# 9. INTERACTIVE DASHBOARD
# --------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Analytics",
    "🚨 At-Risk Students",
    "👤 Student Explorer",
    "🧠 Score Methodology"
])


with tab1:
    left, right = st.columns(2)

    with left:
        st.subheader("Risk Distribution")

        risk_counts = (
            filtered["risk_level"]
            .value_counts()
            .reindex(["High", "Medium", "Low"], fill_value=0)
            .rename_axis("Risk Level")
            .reset_index(name="Students")
        )

        fig = px.bar(
            risk_counts,
            x="Risk Level",
            y="Students",
            color="Risk Level",
            color_discrete_map={
                "High": "#d9534f",
                "Medium": "#f0ad4e",
                "Low": "#36a269"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Average Score by Department")

        department_scores = (
            filtered.groupby("department", as_index=False)
            ["success_score"].mean()
        )

        fig = px.bar(
            department_scores,
            x="department",
            y="success_score",
            labels={
                "department": "Department",
                "success_score": "Average Success Score"
            }
        )

        fig.update_yaxes(range=[0, 100])

        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.subheader("Attendance vs CGPA")

        fig = px.scatter(
            filtered,
            x="attendance",
            y="cgpa",
            color="risk_level",
            hover_name="name",
            hover_data=["student_id", "success_score"],
            color_discrete_map={
                "High": "#d9534f",
                "Medium": "#f0ad4e",
                "Low": "#36a269"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Student Segments")

        segments = (
            filtered["segment"]
            .value_counts()
            .rename_axis("Segment")
            .reset_index(name="Students")
        )

        fig = px.pie(
            segments,
            names="Segment",
            values="Students",
            hole=0.4
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key Insights")

    high_count = int(
        (filtered["risk_level"] == "High").sum()
    )

    if high_count > 0:
        st.warning(
            f"{high_count} student(s) need priority review. "
            "Faculty should verify their individual circumstances."
        )
    else:
        st.success("No high-risk students in the selected view.")

    averages = {
        "Attendance": filtered["attendance"].mean(),
        "Assignment completion": filtered["assignment_completion"].mean(),
        "Placement readiness": filtered[
            ["aptitude_score", "coding_score", "mock_interview_score"]
        ].mean().mean(),
        "Technical skills": filtered["technical_skills"].mean()
    }

    weakest = min(averages, key=averages.get)

    st.write(
        f"*Lowest average indicator:* {weakest} "
        f"({averages[weakest]:.1f}/100)."
    )


with tab2:
    st.subheader("Students Requiring Faculty Review")

    at_risk = filtered[
        filtered["risk_level"].isin(["High", "Medium"])
    ].sort_values("success_score")

    if at_risk.empty:
        st.success("No high- or medium-risk students found.")
    else:
        columns = [
            "student_id", "name", "department",
            "success_score", "risk_level",
            "risk_factors", "suggested_action"
        ]

        st.dataframe(
            at_risk[columns],
            use_container_width=True,
            hide_index=True
        )

        csv_data = at_risk[columns].to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download At-Risk Report",
            data=csv_data,
            file_name="at_risk_students.csv",
            mime="text/csv"
        )


with tab3:
    st.subheader("Individual Student Profile")

    student_options = (
        filtered["student_id"].astype(str)
        + " — "
        + filtered["name"].astype(str)
    )

    selected_student = st.selectbox(
        "Select a student",
        student_options.tolist()
    )

    student_id = selected_student.split(" — ")[0]

    student = filtered[
        filtered["student_id"].astype(str) == student_id
    ].iloc[0]

    a, b, c = st.columns(3)

    a.metric(
        "Success Score",
        f"{student['success_score']:.1f}/100"
    )

    b.metric("Risk Level", student["risk_level"])

    c.metric("Department", student["department"])

    st.write(f"*Name:* {student['name']}")
    st.write(f"*Student ID:* {student['student_id']}")
    st.write(f"*Academic Year:* {int(student['year'])}")
    st.write(f"*Student Segment:* {student['segment']}")

    placement_score = (
        student["aptitude_score"]
        + student["coding_score"]
        + student["mock_interview_score"]
    ) / 3

    indicators = pd.DataFrame({
        "Indicator": [
            "Academic Performance",
            "Attendance",
            "Assignment Completion",
            "LMS Activity",
            "Placement Readiness",
            "Technical Skills",
            "Soft Skills"
        ],
        "Score": [
            student["cgpa"] * 10,
            student["attendance"],
            student["assignment_completion"],
            min(student["lms_logins_week"] / 10, 1) * 100,
            placement_score,
            student["technical_skills"],
            student["soft_skills"]
        ]
    })

    fig = px.bar(
        indicators,
        x="Score",
        y="Indicator",
        orientation="h",
        range_x=[0, 100],
        text_auto=".1f"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Why Was This Student Flagged?")

    st.write(student["risk_factors"])

    st.subheader("Suggested Support Plan")

    st.write(student["suggested_action"])

    st.caption(
        "Suggestions are generated from visible rules. "
        "Faculty should verify the situation with the student."
    )


with tab4:
    st.subheader("How the Success Score Is Calculated")

    methodology = pd.DataFrame({
        "Indicator": [
            "Academic Performance",
            "Attendance",
            "LMS and Assignments",
            "Engagement",
            "Placement Readiness",
            "Technical and Soft Skills",
            "Student Feedback"
        ],
        "Weight (%)": [30, 20, 15, 10, 15, 7, 3]
    })

    st.dataframe(
        methodology,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("""
    *Scoring formula*

    - Academic performance: CGPA converted to a 0–100 scale.
    - Attendance: attendance percentage.
    - LMS and assignments: assignment completion combined with weekly LMS activity.
    - Engagement: events and campus activities.
    - Placement readiness: average aptitude, coding and mock-interview scores.
    - Skills: average technical and soft skills.
    - Feedback: student feedback rating converted to a 0–100 scale.

    *Prototype risk rules*

    - *High risk:* academic concern together with low attendance, or score below 45.
    - *Medium risk:* score below 60, or a key academic, attendance or placement flag.
    - *Low risk:* none of the above rules are triggered.

    *Important:* These are demonstration thresholds, not validated predictions.
    Real deployment requires historical validation, appropriate data permissions,
    privacy safeguards and human review.
    """)


# --------------------------------------------------
# 10. EXPORT ALL STUDENT RESULTS
# --------------------------------------------------

st.divider()

st.subheader("📥 Export Student Analytics")

export_columns = [
    "student_id", "name", "department", "year",
    "cgpa", "attendance", "success_score",
    "risk_level", "risk_factors", "segment",
    "suggested_action"
]

export_csv = filtered[export_columns].to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Complete Student Report",
    data=export_csv,
    file_name="student_success_report.csv",
    mime="text/csv"
)

st.caption(
    "HacXLerate 2026 | FRESH BYTES | Student Success AI"
)

