window.onscroll = function() {
    var scrollTop = window.scrollY; // 获取当前滚动位置
    var content = document.querySelector('.content');
    content.style.transform = 'translateX(' + scrollTop + 'px)'; // 向右滑动
};