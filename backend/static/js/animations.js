document.addEventListener("DOMContentLoaded", () => {
    const loader = document.getElementById("page-loader");
    const curtain = document.getElementById("page-curtain");

    // Show loader on navigation
    document.querySelectorAll("a").forEach(link => {
        if (link.getAttribute("target") !== "_blank" && link.href) {
            link.addEventListener("click", e => {
                if (!link.href.includes("#")) {
                    curtain.classList.add("active");
                    setTimeout(() => loader.classList.remove("hidden"), 200);
                }
            });
        }
    });

    // Show loader on form submit
    document.querySelectorAll("form").forEach(form => {
        form.addEventListener("submit", () => {
            curtain.classList.add("active");
            setTimeout(() => loader.classList.remove("hidden"), 200);
        });
    });

    // Hide loader & curtain on page load
    window.addEventListener("load", () => {
        loader.classList.add("hidden");
        setTimeout(() => curtain.classList.remove("active"), 600);
    });

    // Row reveal effect
    const revealRows = document.querySelectorAll(".reveal-row");
    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add("visible");
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });
    revealRows.forEach(row => observer.observe(row));

    // Tilt effect
    document.querySelectorAll(".tilt-card").forEach(card => {
        card.addEventListener("mousemove", e => {
            const rect = card.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width - 0.5;
            const y = (e.clientY - rect.top) / rect.height - 0.5;
            card.style.transform = `rotateX(${y * 10}deg) rotateY(${x * 10}deg)`;
        });
        card.addEventListener("mouseleave", () => {
            card.style.transform = `rotateX(0) rotateY(0)`;
        });
    });

    // Particle background only on specific pages
    const page = document.body.dataset.page;
    if (page === "app_blueprint.home_page") {
        particlesJS.load('particles-js', '/static/js/particles-config.json', () => {
            console.log("Particles loaded");
        });
    }
});
