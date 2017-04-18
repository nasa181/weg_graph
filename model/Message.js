var mongoose = require('mongoose');
var mongoose_delete = require('mongoose-delete');

var Schema = mongoose.Schema;

var messageSchema = new Schema({
    message: String,
    type: String,
    vec: String,
    vec_from_word: String,
    vec_from_base_line : String,
    date: { type: Date},
    tmp_type: String,
    tmp_type_from_word: String
},{
    timestamps: true,
    collection: 'messages'
});

messageSchema.plugin(mongoose_delete, {
    deletedBy: true,
    deletedAt: true,
    overrideMethods: true
});

var Message = mongoose.model('Message', messageSchema);

module.exports = Message;