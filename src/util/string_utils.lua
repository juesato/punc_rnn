function trimString( s )
   return string.match( s,"^()%s*$") and "" or string.match(s,"^%s*(.*%S)" )
end

function string:trim()
   return string.match(self,"^()%s*$") and "" or string.match(self,"^%s*(.*%S)" )
end

function testTrimString()
	local untrimmedString = "   Some untrimmed string here.     "
	print( "["..untrimmedString.."]" )

	local trimmedString = trimString( untrimmedString )
	print( "["..trimmedString.."]" )
end

function string:split( inSplitPattern, outResults )
   if not outResults then
      outResults = {}
   end
   local theStart = 1
   local theSplitStart, theSplitEnd = string.find( self, inSplitPattern, theStart )
   while theSplitStart do
      table.insert( outResults, string.sub( self, theStart, theSplitStart-1 ) )
      theStart = theSplitEnd + 1
      theSplitStart, theSplitEnd = string.find( self, inSplitPattern, theStart )
   end
   table.insert( outResults, string.sub( self, theStart ) )
   return outResults
end

function containsAlphanumerics( s )
	if string.match(s, "[a-zA-Z0-9]") then
		return true
	else
		return false
	end
end

function testContainsAlphanumerics()
	local s1 = "'s"
	local s2 = '.@#()*@(8@#)($*@'
	local s3 = 'skdhfdkh'
	local s4 = '(@#*&'
	print(s1, containsAlphanumerics(s1))
	print(s2, containsAlphanumerics(s2))
	print(s3, containsAlphanumerics(s3))
	print(s4, containsAlphanumerics(s4))
end

function testStringSplit()
	local myString = "Flintstone, Fred, 101 Rockledge, Bedrock, 98775, 555-555-1212"

	local myTable = myString:split(", ")
	for i = 1,#myTable do
	   print( myTable[i] )
	end
end

function testStringSplit2()
	local data = [==[Two months after a hurricane mauled this country, it is still unclear
how many Hondurans died in the storm, but the government has been
forced to retreat from its earlier conclusion that at least 7,000
people perished. Officials here acknowledge that their initial death
tolls, gathered from panicked local officials in the chaotic days
just after the storm passed, were riddled with inaccuracies. In some
cases, local officials assumed that hundreds had died because entire
neighborhoods had been destroyed, but later learned that the vast
majority of villagers in those places had sought higher ground and
had survived. In early December, the government cut its official death
toll from the storm by close to a quarter _ from 7,007 down to 5,657
_ and suspended the governor of Santa Barbara Province for allegedly
inflating the casualty numbers in that state. The accuracy of even
the smaller figure is in doubt, however, because it includes at least
2,600 deaths reported by distraught family members but never confirmed,
said the man in charge of the government's numbers, Arturo Corrales.
Even the 3,000 confirmed deaths were not based on a body count. The
hurricane moved across Honduras in late October and early November,
dumping record amounts of rain as it disintegrated and causing widespread
flooding across the country. Though the winds quickly abated, floods
and landslides wiped away parts of many towns and villages and severed
communications and wrecked the road network. With journalists raising
questions about the death toll and with international donors pumping
hundreds of millions of dollars in disaster relief into Honduras,
President Carlos Flores has ordered a review of all damages attributed
to the storm. Problems with counting the victims of natural disasters
are common in Third World countries, where many local officials are
poorly trained and illiterate. More than 250 medical and engineering
students have been enlisted for the review and are being trained to
conduct a survey in the worst-hit towns. With help from United Nations
human rights workers and equipment provided by the United States,
the students are fanning out in small groups to try to compile accurate
assessments not only of deaths but also of damage to crops, roads
and bridges, officials said. Flores has promised that they will complete
the survey and that the government will produce an accurate count
by mid-January. In mid-December, the mayors of small towns and villages
were called to emergency meetings in regional capitals to discuss
the death toll. The government has ordered these officials to compile
new lists of the dead and missing, complete with identification numbers
and names. ``We want to straighten the numbers out because there is
no excuse not to have the right numbers anymore,'' said Col. Jorge
Andino Almendares, who is overseeing the government relief operation
in the northeastern part of the country, one of the hardest-hit areas,
after meeting Dec. 16 with more than 40 local mayors in San Pedro
Sula. ``Today we asked for the names of the dead. It has to go down
to the village level.'' But even in the absence of a complete study,
it has become clear that the initial estimates of the dead were not
supported by a count of bodies. In the municipality of El Progreso,
for instance, in Yoro Province, local officials at first reported
that at least 100 people had died in floods and landslides in their
jurisdiction, which includes about 54 villages along the Ulua River.
]==]
	local myTable = data:split(" ")
	for i = 1,#myTable do
	   print( myTable[i] )
	end
end

function testTrimString2()
	local s = '   ldsjfldkjflks   sdfdsfds   '
	print (s:trim() .. '<EOS>')
end

-- testStringSplit2()
-- testStringSplit()
-- testTrimString()
-- testTrimString2()
-- testContainsAlphanumerics()