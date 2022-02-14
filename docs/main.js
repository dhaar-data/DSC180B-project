document.addEventListener("scroll", function () {
    if (document.documentElement.scrollTop > 50) {
        document.getElementsByTagName('nav')[0].classList.value = 'navbar-fixed-top';
        document.getElementById('main-body').style.paddingTop = '100px'
    }
  
    if (document.documentElement.scrollTop < 51) {
        document.getElementsByTagName('nav')[0].classList.value = '';
        document.getElementById('main-body').style.paddingTop = '50px'
    }
});