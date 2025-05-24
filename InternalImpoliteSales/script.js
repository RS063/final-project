
document.addEventListener('DOMContentLoaded', function() {
  // 為技能列表添加hover效果
  const skills = document.querySelectorAll('.skills li');
  skills.forEach(skill => {
    skill.addEventListener('mouseover', function() {
      this.style.backgroundColor = '#e8f0fe';
    });
    skill.addEventListener('mouseout', function() {
      this.style.backgroundColor = '#f8f9fa';
    });
  });
});
