var form = document.getElementById('user-input');
// $SCRIPT_ROOT = {{ request.script_root|tojson|safe }};

form.addEventListener('submit', function(event) {
    event.preventDefault();
});

// $(function() {
//     $("#gen").click(function (event) {
//         $.getJSON($SCRIPT_ROOT + '/_background', {
//                 seed: $('input[name="seed"]'),
//                 words: $('input[name="count"]'),
//                 temp: $('input[name="temp"]')
//             }, function(data) {
//             $("#result").text(data.result);
//         });
//         return false;
//     });
// });