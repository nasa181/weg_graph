var express = require('express');
var router = express.Router();
var Message = require('../model/Message.js');
var graph = require('fbgraph');
var utf8 = require('utf8');
var w2v = require( 'word2vec' );
// var bodyParser = require('body-parser');
var stringUrl = "101655275698/posts?fields=message&limit=100&__paging_token=enc_AdCugpGpdXLsmZAW7XRUZAZC8fLTkUCSxZCV0mK1da6LTNXLTDfVqTtDRyI9avx2UdXVZCpguzhXaIac4vYLsAI6TjXDJqz6lJvSQTfnaVVs8ZAGFYGwZDZD&until=1475664720";
var stringAccessToken = "EAACEdEose0cBAAhCTwgrumQ9AS6upPnntEAf8PnDWL67h94tast5na8Ei5jiYOfeAvm3cXkWK02cA7vhKpCyrdOU2gy3W22r0xpaPYOviVJC0NQhjHyZCMvPsAGLkIghuDZCU4mxQXkEdKSdjgJM8kSt0ZACqYZCMxjNHakHwwZDZD";
/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', { title: 'Express' });
});

router.get('/call_data_from_graph',function(req,res){
    graph.setVersion("2.8");
    graph.setAccessToken(stringAccessToken);
    graph.batch([
    {
    	method: "GET",
    	relative_url: stringUrl // Get the current user's profile information
    }],
    	function(err,mydata) {
    		// res.send(mydata);
	        // var jsonData = decode(mydata[0].body).posts.data;
	        // var text = '';
	        // for (var i in jsonData){
	        // 	text += jsonData[i].message + "</br></br></br></br></br>";
	        // }
	        // res.send(text);


	        var jsonData = decode(mydata[0].body).data;
	        var text = '';
	        for (var i in jsonData){
	        	text += jsonData[i].message + "</br></br></br></br></br>";
	        }
	        res.send(text);


	     //    var jsonData = decode(mydata[0].body).data;
		    // for(var i in jsonData){
		    //     var message = new Message();
		    //     message.message = jsonData[i].message;
		    //     message.type = 'tmp';
		    //     message.save().then(
		    //     	function(message){
		    //     		res.send(message.message + " <<<>>> " + message.type);
		    //     	},function(err){
		    //     		res.send(err);
		    //     	}
		    //     );
	     //    }
    	}
    );
});

function decode(str) {
	return JSON.parse(str);
}

// router.get('/get_rule',function(req,res){
// 	res.send({
// 		'news >>> ข่าว ประกาศผล ประวัติ บทความต่างๆ เรื่องเล่า </br></br></br></br></br>'
// 		'review >>> โชว์ของสวยๆงามๆ รีวิว บอกข้อดีข้อเสีย บอกว่าชอบไม่ชอบ มีความรู้สึกส่วนตัว </br></br></br></br></br>'
// 		'advertisement >>> โฆษณาที่เชิญชวนให้เสียเงิน ส่วนลดของสินค้าบริการต่างๆ </br></br></br></br></br>'
// 		'event >>> เชิญชวนให้ร่วมกิจกรรม รนรงค์ บอกถึงสถานที่จัดงานชัดเจน'
// 	});
// });

router.get('/get_data',function(req,res){
	Message.find().then(
		function(message){
			res.send(message);
		},function(err){
			res.send(err);
		}
	);
});

router.get('/get_data/edit_data',function(req,res){
	Message.find({
		type: { $nin: ['news', 'review', 'advertisement', 'event'] }
	}).then(
		function(message){
			// res.send(message);
			res.render('edit_data', { messages: message });
		},function(err){
			res.send(err);
		}
	);
});

router.post('/get_data/edit_data',function(req,res){
	// console.log("enter");
	Message.findOne({
		_id:req.body.id
	})
	.then(
		function(message){
			// console.log(message);
			message.message = message.message;
			message.type = req.body.type;
			message.save().then(function(message){
				                res.redirect('/get_data/edit_data');
				            },function(err){
				                res.status(400).send({
									success: false,
					                status: 'updata data fail',
					                message: 'updata data fail'
								});
				            });
		},
		function(error){
			res.status(400).send({
				success: false,
                status: 'Bad request',
                message: 'There is no this kind of message'
			});
		}
	);
});

router.get('/try_word2vec',function(req,res){
	var output = '';
	var input = '';
	Message.find()
	.then(
		function(messages){
			for (var i in messages){
				input += messages[i].message;
			}
			// console.log("going to word2phrases");
			// w2v.word2phrases(input,output,function(err){
			// 	console.log("finished word2phrases");
			// 	if(err) res.send(err);
			// 	else res.send(output);
			// })
			console.log("going to word2vec");
			
			w2v.word2vec(input,'output.txt',function(error,output){
				console.log("finished word2vec");
				res.send(output);
			})
		}
	)
});
module.exports = router;
