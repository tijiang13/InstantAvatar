$(window).on('load', function() {
  $('.preload').attr('src', function(i, a){
    $(this).attr('src', '').removeClass('preload').attr('src', a);
  });
});
