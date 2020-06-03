var isGPT = 1;
const simplePopover = "The seed text may be left blank. This model can not recognize any special tags.";
const gptPopover = "The seed text may be left blank. For better results try marking choruses and verses.<br />"
+ "For Example:<br />[Verse 1]<br />My first line<br />Wow, second line<br />Here we go<br />Available tags:<br />[Intro], [Verse #], [Chorus], "
        + "[Pre-Chorus], [Post-Chorus], [Bridge], [Interlude], [Prelude]<br />You may also specify which artist " +
    "should be singing that section by " +
    "adding a colon followed by the artist name.<br />For example:<br />[Intro: Artist Name]";

function update_form(id) {
    var extraOptions = document.getElementById("gptExtraOptions");
    var infoPopover = document.getElementById("infoPopover");
    var wordCount = document.getElementById("count");
    if(id == "simple"){
        extraOptions.style.display = "none";
        $('#infoPopover').attr("data-content", simplePopover);
        $('#infoPopover').popover('hide');
        wordCount.placeholder = "";
        isGPT = 0;
    }
    else if(id == "gpt") {
        extraOptions.style.display = "flex";
        $('#infoPopover').attr("data-content", gptPopover);
        $('#infoPopover').popover('hide');
        wordCount.placeholder = "0 will generate entire song";
        isGPT = 1;
    }
}

function generate_lyrics() {
    const spinnerHTML = "<span class=\"spinner-border spinner-border-sm spinner-size\" role=\"status\"aria-hidden=\"true\"" +
        " id=\"spinner\"></span>";
    var seed = document.getElementById("seed");
    var count = document.getElementById("count");
    var temp = document.getElementById("sliderValue");

    var button = document.getElementById("gen");
    var lyrics = document.getElementById("lyrics");
    var generatedTitle = document.getElementById("generated-title")
    var genArtist = document.getElementById("generated-artist");

    gen.disabled = true;
    gen.innerHTML = spinnerHTML + " Generating...";

    if(isGPT === 1) {
        var artist = document.getElementById("artist");
        var title = document.getElementById("title");
        var topk = document.getElementById("topk")
        var entry = {
            gpt: isGPT,
            seed: seed.value,
            count: count.value,
            temp: temp.value,
            topk: topk.value,
            artist: artist.value,
            title: title.value
        };
    }
    else {
        var entry = {
            gpt: isGPT,
            seed: seed.value,
            count: count.value,
            temp: temp.value
        };
    }
//$(window.origin) does not work
    fetch('/background', {
        method: 'POST',
        credentials: "include",
        body: JSON.stringify(entry),
        cache: "no-cache",
        headers: new Headers({
            "content-type": "application/json"
        })
    })
        .then(function(response) {
            if(response.status !== 200) {
                console.log("Response status not 200 $(response.status)");
                gen.innerHTML = "Generate";
                gen.disabled = false;
                return;
            }
            response.json().then(function (data) {
                generatedTitle.innerText = data.title;
                genArtist.innerText = data.artist;
                lyrics.innerText = data.lyrics;
                gen.innerHTML = "Generate";
                gen.disabled = false;
            })
        })
}