const API_URL = "http://localhost:5000";

/* -------- REGISTER -------- */
const registerForm = document.getElementById("registerForm");
if (registerForm) {
  registerForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const data = {
      fullName: document.getElementById("fullName").value,
      phone: document.getElementById("phone").value,
      email: document.getElementById("email").value,
      password: document.getElementById("password").value
    };

    try {
      const res = await fetch(`${API_URL}/api/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
      });

      const result = await res.json();
      document.getElementById("registerMsg").innerText = result.message;

      if (res.status === 201) {
        registerForm.reset();
      }
    } catch (err) {
      document.getElementById("registerMsg").innerText = "Server error";
    }
  });
}

/* -------- LOGIN -------- */
const loginForm = document.getElementById("loginForm");
if (loginForm) {
  loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const data = {
      email: document.getElementById("loginEmail").value,
      password: document.getElementById("loginPassword").value
    };

    try {
      const res = await fetch(`${API_URL}/api/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
      });

      const result = await res.json();

      if (res.status === 200) {
        // ✅ FIXED HERE
        localStorage.setItem("username", result.user.fullName);
        window.location.href = "home.html";
      } else {
        document.getElementById("loginMsg").innerText = result.message;
      }
    } catch (err) {
      document.getElementById("loginMsg").innerText = "Server error";
    }
  });
}
