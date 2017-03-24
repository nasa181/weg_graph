var mongoose = require('mongoose');
var mongoose_delete = require('mongoose-delete');

var Schema = mongoose.Schema;

var messageSchema = new Schema({
    message: String,
    type: String,
    vec: String,
    date: { type: Date}
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