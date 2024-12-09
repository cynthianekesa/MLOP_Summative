// Wait until the DOM is fully loaded before running the script
document.addEventListener("DOMContentLoaded", function() {
    // Animation for the boxes when the page loads
    const boxes = document.querySelectorAll(".boxes div");
    boxes.forEach((box, index) => {
        box.style.animation = `fadeIn 1s ease ${index * 0.5}s forwards`;
    });

    // Add hover effect to boxes
    boxes.forEach((box) => {
        box.addEventListener("mouseenter", function() {
            box.style.transform = "scale(1.1)";
            box.style.transition = "transform 0.3s ease";
        });
        box.addEventListener("mouseleave", function() {
            box.style.transform = "scale(1)";
        });
    });
});
