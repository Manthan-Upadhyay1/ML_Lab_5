import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Single Layer Perceptron Lab",
    page_icon="🧠",
    layout="wide"
)

# -------------------- Perceptron from the lab notebook --------------------
class Perceptron:
    def __init__(self, n_inputs, lr=0.1, epochs=100, seed=42):
        rng = np.random.default_rng(seed)
        self.w = rng.uniform(-0.5, 0.5, n_inputs)
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs
        self.history = []
        self.converged_at = None

    @staticmethod
    def activation(z):
        return 1 if z >= 0 else 0

    def predict_one(self, x):
        return self.activation(np.dot(self.w, x) + self.b)

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

    def fit(self, X, y):
        for epoch in range(1, self.epochs + 1):
            errors = 0
            for xi, target in zip(X, y):
                output = self.predict_one(xi)
                error = target - output
                if error != 0:
                    self.w = self.w + self.lr * error * xi
                    self.b = self.b + self.lr * error
                    errors += 1
            self.history.append((epoch, self.w.copy(), self.b, errors))
            if errors == 0 and self.converged_at is None:
                self.converged_at = epoch
                break
        return self

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y) * 100


def make_boundary_plot(model, X, y, title, x_label="x1", y_label="x2"):
    fig, ax = plt.subplots(figsize=(6, 4))
    for cls, marker, color in [(0, "o", "tab:red"), (1, "s", "tab:green")]:
        ax.scatter(
            X[y == cls, 0], X[y == cls, 1],
            marker=marker, s=110, color=color, label=f"Class {cls}"
        )
    xs = np.linspace(-0.5, 1.5, 150)
    if abs(model.w[1]) > 1e-9:
        ys = -(model.w[0] * xs + model.b) / model.w[1]
        ax.plot(xs, ys, "k--", label="Decision boundary")
    elif abs(model.w[0]) > 1e-9:
        ax.axvline(-model.b / model.w[0], color="black", linestyle="--",
                   label="Decision boundary")
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def metric_card(label, value):
    st.metric(label, value)


# -------------------- Sidebar --------------------
st.sidebar.title("⚙️ Model settings")
learning_rate = st.sidebar.slider("Learning rate", 0.01, 1.00, 0.10, 0.01)
epochs = st.sidebar.slider("Maximum epochs", 10, 500, 100, 10)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

st.sidebar.divider()
st.sidebar.caption(
    "The same Perceptron learning rule used in the lab notebook is applied "
    "throughout this app."
)

# -------------------- Header --------------------
st.title("🧠 Single Layer Perceptron")
st.write(
    "An interactive lab dashboard for understanding binary classification, "
    "decision boundaries, convergence, weights, and bias."
)
st.info(
    "A perceptron predicts 0 or 1 using a weighted sum of its inputs. "
    "It updates its weights and bias whenever a training example is misclassified."
)

X_gates = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
gates = {
    "AND": np.array([0, 0, 0, 1]),
    "OR": np.array([0, 1, 1, 1]),
    "NAND": np.array([1, 1, 1, 0]),
    "XOR": np.array([0, 1, 1, 0]),
}

tab1, tab2, tab3, tab4 = st.tabs([
    "🔌 Logic gates",
    "❌ XOR experiment",
    "🎓 Scholarship eligibility",
    "👨‍💻 Candidate selection"
])

# -------------------- Tab 1: AND, OR, NAND --------------------
with tab1:
    st.subheader("Question 1: Train a perceptron for AND, OR and NAND")
    st.write("These three gates are linearly separable, so a single-layer perceptron can learn them.")

    selected_gate = st.selectbox("Choose a logic gate", ["AND", "OR", "NAND"])
    y_gate = gates[selected_gate]
    model = Perceptron(2, learning_rate, epochs, int(seed)).fit(X_gates, y_gate)
    predictions = model.predict(X_gates)

    truth_table = pd.DataFrame({
        "x1": X_gates[:, 0],
        "x2": X_gates[:, 1],
        "Actual": y_gate,
        "Predicted": predictions,
        "Correct": np.where(predictions == y_gate, "Yes", "No")
    })
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{model.accuracy(X_gates, y_gate):.1f}%")
    c2.metric("Converged at epoch", str(model.converged_at or "Not converged"))
    c3.metric("Misclassified samples", int(np.sum(predictions != y_gate)))

    left, right = st.columns([1, 1.2])
    with left:
        st.markdown("**Truth table and predictions**")
        st.dataframe(truth_table, use_container_width=True, hide_index=True)
        st.markdown("**Learned parameters**")
        st.write(f"Weights: `{np.round(model.w, 3).tolist()}`")
        st.write(f"Bias: `{model.b:.3f}`")
    with right:
        st.pyplot(make_boundary_plot(
            model, X_gates, y_gate,
            f"{selected_gate} gate decision boundary"
        ), use_container_width=True)

    st.markdown("**Compare all three gates**")
    comparison = []
    for name in ["AND", "OR", "NAND"]:
        fitted = Perceptron(2, learning_rate, epochs, int(seed)).fit(X_gates, gates[name])
        comparison.append({
            "Logic Gate": name,
            "Linearly Separable": "Yes",
            "Converged at Epoch": fitted.converged_at or "Never",
            "Accuracy (%)": round(fitted.accuracy(X_gates, gates[name]), 1),
            "Learning outcome": "Converges" if fitted.accuracy(X_gates, gates[name]) == 100 else "Not perfect"
        })
    st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)

# -------------------- Tab 2: XOR --------------------
with tab2:
    st.subheader("Question 2: Why does a single-layer perceptron fail on XOR?")
    st.write(
        "XOR is not linearly separable. The experiment below shows the errors "
        "across epochs and the resulting predictions."
    )
    xor_model = Perceptron(2, learning_rate, min(epochs, 500), int(seed)).fit(
        X_gates, gates["XOR"]
    )
    xor_pred = xor_model.predict(X_gates)
    xor_table = pd.DataFrame({
        "x1": X_gates[:, 0],
        "x2": X_gates[:, 1],
        "Actual": gates["XOR"],
        "Predicted": xor_pred,
        "Correct": np.where(xor_pred == gates["XOR"], "Yes", "No")
    })

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{xor_model.accuracy(X_gates, gates['XOR']):.1f}%")
    c2.metric("Converged at epoch", str(xor_model.converged_at or "Never"))
    c3.metric("Final epoch errors", xor_model.history[-1][3])

    left, right = st.columns(2)
    with left:
        st.markdown("**XOR predictions**")
        st.dataframe(xor_table, use_container_width=True, hide_index=True)
        st.write(f"Weights: `{np.round(xor_model.w, 3).tolist()}`")
        st.write(f"Bias: `{xor_model.b:.3f}`")
    with right:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(
            [h[0] for h in xor_model.history],
            [h[3] for h in xor_model.history],
            marker="o"
        )
        ax.set_title("XOR: misclassified samples per epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Misclassified samples")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    st.pyplot(make_boundary_plot(
        xor_model, X_gates, gates["XOR"],
        "XOR: one straight line cannot separate both classes"
    ), use_container_width=True)
    st.warning(
        "The single-layer perceptron cannot reach a perfect XOR solution because "
        "no single straight decision boundary separates the two classes. "
        "A Multi-Layer Perceptron with hidden layers is needed."
    )

    with st.expander("View weight updates across epochs"):
        history_rows = [{
            "Epoch": h[0],
            "Weights": str(np.round(h[1], 3).tolist()),
            "Bias": round(h[2], 3),
            "Misclassified": h[3]
        } for h in xor_model.history]
        st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)

# -------------------- Tab 3: Scholarship dataset --------------------
with tab3:
    st.subheader("Question 3A: Scholarship eligibility")
    st.write(
        "The lab dataset uses Internal Assessment and Attendance (%) to predict "
        "Scholarship Eligibility."
    )
    df_a = pd.DataFrame({
        "Student": [f"S{i}" for i in range(1, 9)],
        "Internal Assessment": [85, 75, 65, 45, 50, 30, 90, 40],
        "Attendance (%)": [90, 85, 80, 60, 55, 45, 95, 50],
        "Scholarship Eligibility": [1, 1, 1, 0, 0, 0, 1, 0],
    })
    Xa_raw = df_a[["Internal Assessment", "Attendance (%)"]].values.astype(float)
    ya = df_a["Scholarship Eligibility"].values
    mn, mx = Xa_raw.min(axis=0), Xa_raw.max(axis=0)
    Xa = (Xa_raw - mn) / (mx - mn)

    scholarship_model = Perceptron(2, learning_rate, max(epochs, 200), int(seed)).fit(Xa, ya)
    scholarship_pred = scholarship_model.predict(Xa)
    result_a = df_a.copy()
    result_a["Predicted"] = scholarship_pred
    result_a["Correct"] = np.where(scholarship_pred == ya, "Yes", "No")

    c1, c2, c3 = st.columns(3)
    c1.metric("Training accuracy", f"{scholarship_model.accuracy(Xa, ya):.1f}%")
    c2.metric("Converged at epoch", str(scholarship_model.converged_at or "Never"))
    c3.metric("Students in dataset", len(df_a))

    st.markdown("**Dataset and model predictions**")
    st.dataframe(result_a, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(
            [h[0] for h in scholarship_model.history],
            [h[3] for h in scholarship_model.history],
            marker="o"
        )
        ax.set_title("Scholarship: errors per epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Misclassified samples")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
    with right:
        fig, ax = plt.subplots(figsize=(6, 4))
        for cls, color, label in [(0, "tab:red", "Not Eligible"), (1, "tab:green", "Eligible")]:
            ax.scatter(
                Xa_raw[ya == cls, 0], Xa_raw[ya == cls, 1],
                color=color, s=90, label=label
            )
        if abs(scholarship_model.w[1]) > 1e-9:
            xs = np.linspace(0, 1, 100)
            ys = -(scholarship_model.w[0] * xs + scholarship_model.b) / scholarship_model.w[1]
            ax.plot(
                xs * (mx[0] - mn[0]) + mn[0],
                ys * (mx[1] - mn[1]) + mn[1],
                "k--", label="Decision boundary"
            )
        ax.set_xlabel("Internal Assessment")
        ax.set_ylabel("Attendance (%)")
        ax.set_title("Scholarship eligibility decision boundary")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    st.markdown("**Try a new student**")
    col1, col2 = st.columns(2)
    with col1:
        new_marks = st.slider("Internal Assessment", 0, 100, 70)
    with col2:
        new_attendance = st.slider("Attendance (%)", 0, 100, 75)
    new_x = np.array([[
        (new_marks - mn[0]) / (mx[0] - mn[0]),
        (new_attendance - mn[1]) / (mx[1] - mn[1])
    ]])
    new_prediction = scholarship_model.predict(new_x)[0]
    if new_prediction == 1:
        st.success("Model prediction: Eligible (1)")
    else:
        st.error("Model prediction: Not Eligible (0)")
    st.caption("This is a prediction from the small lab dataset, not an official scholarship decision.")

# -------------------- Tab 4: Candidate selection --------------------
with tab4:
    st.subheader("Question 3B: Candidate selection")
    st.write(
        "The model uses two binary inputs, Technical Test and Interview, "
        "to predict the Selection label from the lab dataset."
    )
    df_b = pd.DataFrame({
        "Candidate": [f"C{i}" for i in range(1, 9)],
        "Technical Test": [1, 1, 0, 0, 1, 0, 1, 0],
        "Interview": [1, 0, 1, 0, 1, 0, 0, 1],
        "Selection": [1, 1, 1, 0, 1, 0, 1, 1],
    })
    Xb = df_b[["Technical Test", "Interview"]].values
    yb = df_b["Selection"].values
    candidate_model = Perceptron(2, learning_rate, epochs, int(seed)).fit(Xb, yb)
    candidate_pred = candidate_model.predict(Xb)
    result_b = df_b.copy()
    result_b["Predicted"] = candidate_pred
    result_b["Correct"] = np.where(candidate_pred == yb, "Yes", "No")

    c1, c2, c3 = st.columns(3)
    c1.metric("Training accuracy", f"{candidate_model.accuracy(Xb, yb):.1f}%")
    c2.metric("Converged at epoch", str(candidate_model.converged_at or "Never"))
    c3.metric("Candidates", len(df_b))
    st.dataframe(result_b, use_container_width=True, hide_index=True)

    st.markdown("**Test a candidate**")
    col1, col2 = st.columns(2)
    with col1:
        technical = st.selectbox("Technical Test", [0, 1], format_func=lambda x: "Passed (1)" if x else "Failed (0)")
    with col2:
        interview = st.selectbox("Interview", [0, 1], format_func=lambda x: "Passed (1)" if x else "Failed (0)")
    candidate_result = candidate_model.predict(np.array([[technical, interview]]))[0]
    if candidate_result == 1:
        st.success("Model prediction: Selected (1)")
    else:
        st.error("Model prediction: Not selected (0)")

    st.markdown("**Learned parameters**")
    st.write(f"Weights: `{np.round(candidate_model.w, 3).tolist()}`")
    st.write(f"Bias: `{candidate_model.b:.3f}`")

# -------------------- Conclusion --------------------
st.divider()
st.subheader("Lab conclusion")
st.markdown("""
- **AND, OR and NAND** are linearly separable, so a single-layer perceptron can learn them.
- **XOR** is not linearly separable. A single-layer perceptron cannot reach a perfect solution.
- **Weights** control the orientation of the decision boundary, while **bias** shifts it.
- Feature scaling helps when working with real-valued inputs, as shown in the scholarship example.
- Non-linearly separable problems such as XOR require a **Multi-Layer Perceptron (MLP)** with hidden layers.
""")
st.caption("Single Layer Perceptron Lab • Built from the supplied Lab 5 notebook")
