$(document).ready(function() {
   $("select").change(function() {
     $("select option:selected").each(function() {
      if ($(this).attr("value") == "Ozero") {
         $(".box").hide();
         $(".white").show();
       } 
      if ($(this).attr("value") == "Oone") {
         $(".box").hide();
         $(".red").show();
       }
       if ($(this).attr("value") == "Otwo") {
         $(".box").hide();
         $(".green").show();
       }
       if ($(this).attr("value") == "Othree") {
         $(".box").hide();
         $(".blue").show();
       }
     });
   }).change();
 });